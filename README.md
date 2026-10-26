# offpeak-router

**A peak-aware, OpenAI-compatible LLM router.** It sends each request to whichever
provider is currently outside its peak-pricing window, and tracks the monthly
subscription allowance so it gets spent where it buys the most tokens.

Single static Go binary. No dependencies. Any client that speaks the OpenAI API can
point at it.

```
your app ──▶ http://127.0.0.1:8788/v1 ──▶ provider A │ provider B │ provider C
                                              │
                                    picked per request by
                                    UTC clock + allowance burn
```

---

## The problem

Several inference providers sell a monthly subscription whose included usage is worth
far more than the price, then charge **double** during a published peak window.

If you have two of them, and their peak windows do not overlap, there is always one
provider running at its cheap rate. A router that knows the schedule can keep nearly
all traffic on the cheap tier — **without changing the model or the client**.

The trap most people hit: the subscription allowance is denominated in *usage
dollars*, not requests. Spending it during peak halves how many requests it covers.
So the allowance itself has to be spent at the right time, not just the traffic.

## How the order is decided

For every request, providers are ranked and the first usable one wins:

| # | rule | why |
|---|------|-----|
| 1 | allowance left **and** off-peak | covers ~2× more requests than the same allowance at peak |
| 2 | other allowance left **and** off-peak | same reasoning |
| 3 | any allowance left, even at peak | a prepaid token is never worse than a metered one |
| 4 | metered: cheapest current marginal cost | e.g. a provider's own top-up vs. a third provider |

A provider priced at zero would sort to the front of the metered tier — free
beats billed. That is the natural place for a free variant if you add one.

On a transport error or HTTP 5xx it falls through to the next candidate, so a dead
provider costs latency rather than a failed request. A provider with no API key is
skipped without an attempt. After 3 consecutive failures a provider is skipped for
60 seconds, then one probe request is let through.

### Measured saving

On a workload that is ~96% prompt-cache reads, routing between two subscriptions
whose peak windows are disjoint:

```
                 monthly bill
one provider      $144.94
offpeak-router     $94.39   (-35%)
```

Numbers depend entirely on your volume, cache-hit ratio and provider mix — measure
your own with `/status` before believing any figure, including this one.

## Quick start

```bash
git clone https://github.com/wzul/offpeak-router
cd offpeak-router
go build -trimpath -ldflags='-s -w' -o offpeak-router .
```

Export the keys for the providers you use, then run:

```bash
export OPENCODE_API_KEY=...
export OLLAMA_API_KEY=...
export NOUS_API_KEY=...
./offpeak-router
```

Point your client at it:

```bash
curl -s http://127.0.0.1:8788/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"anything","messages":[{"role":"user","content":"hello"}]}'
```

The `model` field selects the model — the router sends each provider the id that
provider uses for it. Ask for `deepseek-v4.1-flash` and OpenCode Go, Ollama and
Nous each receive their own spelling; ask for a model only one of them has and
only that one is tried. An unknown model is a `400` listing what exists, never a
silent substitution. Response headers tell you what happened:

```
X-Router-Provider: ollama
X-Router-Model: deepseek-v4.1-flash
X-Router-Usd: 0.002069
X-Router-Failover: false
```

### Models and endpoint families

`GET /v1/models` lists every canonical model, the providers that can serve it,
and the endpoint family it speaks:

```json
{"id": "clef", "api": "systemone", "providers": ["ollama"]}
```

A decision model answers on `/v1/systemone` — one state plus a schema of typed
questions, no chat messages — so it is refused on `/v1/chat/completions` and
vice versa. Both routes share the same auth, failover and billing:

```bash
curl -s http://127.0.0.1:8788/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"clef","state":"I was charged twice, please refund me.",
       "questions":{"refund":{"type":"noul","instructions":"Is a refund requested?"}}}'
```

A provider can price its models apart, which matters when one provider hosts both
a cheap chat model and an expensive one:

```json
"models": {
  "deepseek-v4.1-flash": {"id": "deepseek-v4.1-flash:1017"},
  "clef": {"id": "clef:27b", "api": "systemone",
           "rates": {"in": 0.24, "cache": 0.24, "out": 0.24, "peak_mult": 1.0}}
}
```

`id` defaults to the canonical name and `rates` to the provider's block, so the
short form `{"model": "x"}` still works for a one-model provider.

### Check the decision without spending anything

```bash
ROUTER_SHADOW=1 ./offpeak-router &
curl -s http://127.0.0.1:8788/status | jq
```

`ROUTER_SHADOW=1` returns the routing decision and never contacts an upstream.
Run it for a day before you move real traffic — it costs nothing and shows exactly
which provider each hour would have used.

### Web search and fetch

The same binary also fronts the web layer, behind the same bearer token:

```bash
curl -s -X POST http://127.0.0.1:8788/v1/search \
  -H "Authorization: Bearer $ROUTER_AUTH_TOKEN" -H 'Content-Type: application/json' \
  -d '{"query":"golang net/http timeout","limit":5}'
# {"success":true,"query":"...","data":{"web":[{"title","url","description","position"}]}}

curl -s -X POST http://127.0.0.1:8788/v1/fetch \
  -H "Authorization: Bearer $ROUTER_AUTH_TOKEN" -H 'Content-Type: application/json' \
  -d '{"url":"https://go.dev/doc/"}'
# {"success":true,"data":[{"url","title","content","status"}]}
```

- **`/v1/search`** takes `{"query","limit"}` (max 20) or `GET /v1/search?q=`. Backend is
  Brave when `BRAVE_SEARCH_API_KEY` is set, otherwise DuckDuckGo's HTML endpoint, which
  needs no key. The response carries `X-Router-Backend`; `/status` reports the active one.
- **`/v1/fetch`** takes `{"url"}` or `{"urls":[…]}` (max 5) or `GET /v1/fetch?url=`. It
  follows up to five redirects, caps the body at 4 MiB, and reduces HTML to a title plus
  readable text (scripts, styles, `<head>` and inline SVG dropped; block tags become line
  breaks). Non-HTML bodies are returned verbatim.
- **Both refuse non-public targets.** Any host that resolves to loopback, RFC1918,
  link-local (including `169.254.169.254`), carrier-grade NAT or multicast is rejected
  *before* a connection is made, and the same check runs on every redirect hop — an
  authenticated route on a host with internal services is otherwise an SSRF pivot.
- Neither route costs anything: they are not model calls and do not touch the spend ledger.
  `/status` counts them under `web.search_calls` / `web.fetch_calls`.

### Capabilities: does this route take images?

Discovery used to answer only "which models exist". A vision model and a text-only model
looked identical, so an agent looking for image support found no evidence of one and
reported that the router has no vision model — while the model behind it answered "Red"
for a red PNG on two upstreams. Every discovery route now states the capability, and only
capabilities that a live request proved:

```bash
curl -s -H "Authorization: Bearer $ROUTER_AUTH_TOKEN" http://127.0.0.1:8788/v1/models
# {"data":[{"id":"deepseek-v4.1-flash","api":"chat","providers":["go","nous","ollama"],
#           "capabilities":["completion","vision","thinking"],
#           "capability_providers":{"vision":["go","ollama"],"thinking":["go","ollama"]},
#           "input_modalities":["text","image"]}]}
```

- **`capability_providers` is the honest part.** Vision was proved on `go` and `ollama`
  with a real PNG; `nous` is absent because it was never verified, not because it failed.
  A client that must not risk a 400 can read which upstream carries the capability.
- **`capabilities` uses Ollama's vocabulary** (`completion`, `vision`, `thinking`, `tools`),
  the same strings `/api/show` returns, so one parser works for both dialects.
  `input_modalities` is there for clients that follow the OpenAI shape instead.
- **Never listed without proof.** `clef` reports `["completion"]` only: it is a decision
  model, and advertising a capability nobody tested is how a client ships a broken path.
- **A model can exist on one provider only.** `gemma4` maps to ollama's `gemma4:31b`;
  opencode's `go` answers `Model is unavailable` for it, so `capability_providers.vision`
  names `ollama` alone. Both the canonical name and the upstream id resolve.

### Model discovery, in both dialects

`/v1/models` is the OpenAI answer. Some clients do not ask it first: they probe for
Ollama (`{server}/api/tags`, expecting a `models` key) to decide whether a server is
usable at all, and a 404 there makes them give up even though `/v1/models` would have
answered. Both shapes are served from the same table:

```bash
curl -s -H "Authorization: Bearer $ROUTER_AUTH_TOKEN" http://127.0.0.1:8788/api/tags
# {"models":[{"name":"deepseek-v4.1-flash:latest","model":"…","modified_at":"…","size":0,
#             "digest":"","details":{"family":"chat","families":["go","nous","ollama"],
#             "router_api":"chat","router_providers":["go","nous","ollama"]}}]}
curl -s -H "Authorization: Bearer $ROUTER_AUTH_TOKEN" http://127.0.0.1:8788/api/version
# {"version":"1.3.0"}
```

- Names follow Ollama's `name:tag` convention; a bare id gets `:latest`. Both `name`
  and `model` carry it because clients read either.
- `size` stays `0` and `digest` stays empty: a routed model has no local weights, and
  inventing a blob description would be a lie a client could act on.
- `details.router_api` is the route family that model actually answers on — `chat` for
  `/v1/chat/completions`, `systemone` for `/v1/systemone`. A decision model listed here
  still rejects a chat request, exactly as `/v1/models` reports it.
- Both routes are behind the bearer token like every other route except `/health`.

### Speaking Ollama's chat dialect

Discovery alone was not enough: a client that found the router through `/api/tags`
then POSTed `/api/chat` and got a 404. Both chat routes now translate in either
direction, with the same bearer token and the same spend ledger:

```bash
# streams NDJSON by default, exactly like Ollama
curl -s -X POST http://127.0.0.1:8788/api/chat \
  -H "Authorization: Bearer $ROUTER_AUTH_TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"hi"}],
       "options":{"num_predict":64,"temperature":0.4}}'
# {"model":"deepseek-v4.1-flash:latest","created_at":"…","done":false,
#  "message":{"role":"assistant","content":"…"}}
# … {"done":true,"done_reason":"stop","prompt_eval_count":41,"eval_count":22}

curl -s -X POST http://127.0.0.1:8788/api/generate \
  -H "Authorization: Bearer $ROUTER_AUTH_TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","prompt":"say hi","stream":false}'
```

- **Routing, failover, breakers and billing are the `/v1` code path.** A client must
  not get different routing because it chose the other dialect; `X-Router-*` headers
  and `/status` counters are identical.
- **Streaming is the default**, as in Ollama. A missing `stream` field means NDJSON,
  so a single JSON document is only ever sent when `stream:false` is explicit.
- **Options are mapped, unknown ones dropped** — `temperature`, `top_p`, `seed`,
  `num_predict`→`max_tokens`, `stop`. `top_k` has no OpenAI equivalent and forwarding
  it earns a 400 that would be blamed on the provider.
- **Reasoning is surfaced as `message.thinking`** (Ollama's field), because an upstream
  sends it as `reasoning_content` and a thinking model with a small budget can finish
  with empty `content` — dropping the reasoning makes a working reply look empty.
- **`format:"json"` sets `response_format:"json_object"`**; images carried as bare
  base64 are turned into data URLs with the media type sniffed from the bytes.
- **A decision model is refused on `/api/chat`** with a 400 naming `/v1/systemone`,
  and an unknown model answers Ollama's own 404 wording.

### Web search and fetch in Ollama Cloud's dialect

Ollama's own API puts these at `/api/web_search` and `/api/web_fetch`, with a different
body from the router's `/v1` routes. A client built on the ollama SDK calls those names
and parses those fields, so both are served from one implementation:

```bash
curl -s -X POST http://127.0.0.1:8788/api/web_search \
  -H "Authorization: Bearer $ROUTER_AUTH_TOKEN" -H 'Content-Type: application/json' \
  -d '{"query":"what is ollama?","max_results":3}'
# {"results":[{"title":"…","url":"…","content":"snippet"}]}

curl -s -X POST http://127.0.0.1:8788/api/web_fetch \
  -H "Authorization: Bearer $ROUTER_AUTH_TOKEN" -H 'Content-Type: application/json' \
  -d '{"url":"https://go.dev/doc/"}'
# {"title":"…","content":"readable text","links":["https://…","https://…"]}
```

- **Ollama's field names, not the router's.** The snippet is `content` (the `/v1` route
  calls it `description`) and there is no `position`. Both spellings are never sent, or a
  caller would start depending on something the vendor does not have.
- **`links` is always an array**, `[]` when the page has none — a client iterating over
  `null` crashes instead of showing an empty list.
- **`max_results` defaults to 5 and is capped at 10**, matching Ollama's own bounds; asking
  for more returns 10 rather than silently behaving better than the vendor.
- **POST only**, like Ollama. A `GET /api/web_search?q=…` answers 405 instead of quietly
  working and making a client ship code the vendor would reject.
- **Links are resolved and deduplicated** against the page they came from, with fragments
  dropped and non-http(s) schemes skipped, capped at 100 so a large site's navigation does
  not bury the content.
- The `/v1/search` and `/v1/fetch` routes keep the gateway's own shape; these are additive.

## Configuration

Works out of the box with three providers of DeepSeek V4.1 Flash. For anything
else, pass a JSON file:

```bash
ROUTER_CONFIG=./config.json ./offpeak-router
```

```json
{
  "listen": "127.0.0.1:8788",
  "ledger": "/var/lib/offpeak-router/spend.json",
  "fallback_order": ["cheap", "backup", "metered"],
  "providers": {
    "cheap": {
      "base_url": "https://opencode.ai/inference/go/openai/v1",
      "model": "deepseek-v4.1-flash",
      "api_key_env": "OPENCODE_API_KEY",
      "monthly_allowance_usd": 80,
      "rates": { "in": 0.15, "cache": 0.003, "out": 0.60, "peak_mult": 2.0 },
      "peak_windows_utc": [[1, 4], [6, 10]]
    },
    "metered": {
      "base_url": "https://inference-api.nousresearch.com/v1",
      "model": "deepseek/deepseek-v4.1-flash-1017",
      "api_key_env": "NOUS_API_KEY",
      "rates": { "in": 0.02, "cache": 0.015, "out": 1.00, "peak_mult": 1.0 }
    }
  }
}
```

| field | meaning |
|---|---|
| `base_url` | OpenAI-compatible root; the router appends `/chat/completions` |
| `api_key_env` | **name** of the environment variable holding the key — never the key itself |
| `monthly_allowance_usd` | included usage per calendar month. Omit for a metered provider |
| `hard_cap` | `true` = the allowance cannot be exceeded, so the provider is dropped once it is spent. Omit if you can top up past the allowance |
| `rates` | USD per 1M tokens, split into fresh input / cache reads / output |
| `peak_mult` | multiplier applied during peak. Use `1.0` for a provider that never doubles |
| `peak_windows_utc` | half-open `[start, end)` UTC hours, **Monday–Friday only**; weekends are fully off-peak |
| `session_header` | `true` = send `x-opencode-session`, a stable per-conversation id. OpenCode Go **rejects** a request without it |
| `user_agent` | override the User-Agent for providers that demand a specific one |

The session id is taken from the client's own `x-opencode-session` when present, so
real session tracking survives the proxy. Otherwise it is derived from the
conversation prefix (system prompt plus first user turn), which stays stable for
every later turn without the router holding any state.

`peak_windows_utc` is the whole point of the project, so get it from your provider's
pricing page and keep it current. An empty list means "never peaks".

### Environment variables

| variable | default | meaning |
|---|---|---|
| `ROUTER_CONFIG` | — | path to the JSON config above |
| `ROUTER_ADDR` | `127.0.0.1:8788` | listen address |
| `ROUTER_DB` | `spend.json` | allowance ledger |
| `ROUTER_SHADOW` | `0` | `1` = decide only, never call upstream |
| `ROUTER_AUTH_TOKEN` | - | when set, every endpoint except `/health` requires a bearer token in the Authorization header |
| `ROUTER_<NAME>_BASE` / `_MODEL` | — | retarget one provider without a config file |

## Endpoints

| path | purpose |
|---|---|
| `POST /v1/chat/completions` | proxy — streaming and non-streaming |
| `POST /v1/systemone` | proxy for decision models (one state + typed schema in) |
| `POST /v1/search` · `GET /v1/search?q=` | web search — Brave or DuckDuckGo, no key required |
| `POST /v1/fetch` · `GET /v1/fetch?url=` | fetch a page and return title + readable text; non-public targets refused |
| `GET /v1/models` | the configured provider model ids |
| `GET /api/tags` | the same models in Ollama's shape (`name:tag`, `details`) — for clients that probe for Ollama before trusting a server |
| `GET /api/version` | Ollama-style version string, so that probe succeeds |
| `POST /api/chat` · `POST /api/generate` | chat in Ollama's dialect — request translated to the OpenAI payload, answer translated back (NDJSON when streaming) |
| `POST /api/show` · `POST /api/ps` | model details from the table; nothing is ever resident locally |
| `POST /api/web_search` | web search in Ollama Cloud's shape — `{query, max_results}` → `{results:[{title,url,content}]}` |
| `POST /api/web_fetch` | page fetch in Ollama Cloud's shape — `{url}` → `{title,content,links[]}` |
| `GET /status` | per provider: peak flag, allowance, spend, remaining, `hard_cap`, whether the key is present — plus the live order and breaker state |
| `GET /health` | liveness, plus which keys are present (never their values) |

## Why Go

It is an always-on sidecar on a small box, so the footprint is worth measuring:

| | this router (Go) | equivalent FastAPI service |
|---|---|---|
| RSS | 9.3 MB | 62.4 MB |
| PSS | 7.8 MB | 51.4 MB |
| open fds | 6 | 113 |
| on-disk | 6.9 MB binary | 59 MB venv |
| runtime deps | none | fastapi, uvicorn, httpx, starlette, pydantic |

The Go numbers are the real reason to pick it: nothing to install, nothing to break on
an upgrade, and one file to copy when you redeploy. The memory saving is a bonus, not
the argument.

## Tests

```bash
go test ./...          # 26 tests
go test -race ./...    # same, under the race detector
go test -v ./...       # per-test detail
```

The suite covers every peak-window boundary for both schedules (00:59/01:00,
03:59/04:00, …, 17:59/18:00), weekend handling, the disjoint-windows invariant, the
routing order under every combination of peak state and exhausted allowance,
failover, circuit breaking, shadow mode, streaming relay, and cost accounting from
the upstream `usage` block.

It also pins the two properties that are easy to lose:

- **A hard-capped provider disappears once its allowance is spent** — it cannot serve at
  any price, so listing it only wastes a round-trip.
- **Nothing in the routing knows a provider by name.** A test loads an arbitrary two-provider
  set and asserts the tiers still work and no built-in name leaks into the result.

The tests are mutation-checked: shifting either peak window by one hour, disabling the
weekend rule, or breaking the circuit breaker in either direction makes the suite fail.

## Known limitations

- **The schedule is UTC and Mon–Fri only.** Holiday calendars are not modelled. If a
  provider treats a public holiday as off-peak, this router will route away from it
  unnecessarily — suboptimal, never unsafe.
- **Spend is estimated, not authoritative.** Allowance is accrued from each response's
  `usage` block. A provider that omits usage on a stream bills as `$0` locally, so the
  ledger drifts low and will not stop you early. Cross-check `/status` against the
  provider's own billing page.
- **Cache tokens come from `prompt_tokens_details.cached_tokens`**, falling back to
  `prompt_cache_hit_tokens`. If a provider reports neither, cached reads bill at the
  full input rate and the estimate drifts high.
- **Single process, single ledger file.** Written atomically (temp file + rename) but
  not designed to be shared over NFS or between concurrent instances.
- **Auth is off unless you set `ROUTER_AUTH_TOKEN`.** It binds to loopback by default.
  If you expose it, set the token — the router holds your provider keys and will spend
  your credit for anyone who can reach it. `/health` is deliberately left open so a
  supervisor can probe liveness without the credential; it only reports whether each
  key is present, never its value.
- **Bearer, not basic.** An OpenAI client sends `api_key` as a bearer
  token in the Authorization header, so bearer is the one scheme that works
  without a custom client. Basic auth would need a proxy in front to
  translate it.

## License

MIT — see [LICENSE](LICENSE).
