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

The `model` field in the request is ignored — the router substitutes the model id
each provider expects. Response headers tell you what happened:

```
X-Router-Provider: ollama
X-Router-Usd: 0.002069
X-Router-Failover: false
```

### Check the decision without spending anything

```bash
ROUTER_SHADOW=1 ./offpeak-router &
curl -s http://127.0.0.1:8788/status | jq
```

`ROUTER_SHADOW=1` returns the routing decision and never contacts an upstream.
Run it for a day before you move real traffic — it costs nothing and shows exactly
which provider each hour would have used.

## Configuration

Works out of the box with three providers of DeepSeek V4.1 Flash. For anything else,
pass a JSON file:

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
| `GET /v1/models` | the configured provider model ids |
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
