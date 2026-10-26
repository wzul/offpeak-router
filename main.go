package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// Version is reported in the User-Agent and should match the release tag.
const Version = "1.12.0"

// DefaultUserAgent lives in headers.go: it carries the client identity the
// upstream expects, because naming this router in the User-Agent is a signpost
// for a third-party provider. An identifiable UA is still required — Cloudflare
// rejects Go's default and an empty one.

// ModelRoute is one model a provider can serve: the id the provider expects,
// plus an optional rate override for providers whose models are priced apart.
type ModelRoute struct {
	ID    string
	Rates *Rates
	// API is the endpoint family the model speaks. Most models are chat
	// completions; a decision model like clef only answers /v1/systemone, and
	// offering it on the chat route would produce an upstream 400 on every call.
	API string
}

// Endpoint families a model can belong to.
const (
	APIChat      = "chat"
	APISystemOne = "systemone"
)

type Provider struct {
	Base string
	// Model is the single canonical model when Models is empty. It is kept so a
	// provider that serves one model stays a one-line entry in a config file.
	Model string
	// Models maps a canonical model name to what this provider calls it. This is
	// the whole point of the table: the clients ask for one stable name and each
	// provider gets its own spelling.
	Models map[string]ModelRoute
	KeyEnv string
	// UA overrides DefaultUserAgent for providers that demand something
	// specific. Empty means DefaultUserAgent.
	UA string
	// OAuth marks a provider whose credential is a short-lived access token minted
	// from a refresh token, not a static key. KeyEnv is then unused.
	OAuth *OAuthConfig
	// SessionHeader sends x-opencode-session, a stable per-conversation id.
	// OpenCode Go rejects a request without one ("MissingSessionID") and uses
	// it to optimise routing and prompt caching.
	SessionHeader bool
}

// conversationID derives a stable identifier for the conversation a request
// belongs to. A client that sends its own x-opencode-session wins, so real
// session tracking is preserved through the proxy; otherwise the id is derived
// from the conversation prefix, which is stable for every later turn of the
// same conversation without the router holding any state.
func conversationID(r *http.Request, body map[string]any) string {
	if h := r.Header.Get("x-opencode-session"); h != "" {
		return h
	}
	h := sha256.New()
	msgs, _ := body["messages"].([]any)
	for _, raw := range msgs {
		m, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		role, _ := m["role"].(string)
		switch role {
		case "system", "developer":
			fmt.Fprintf(h, "%s\x00%v\x00", role, m["content"])
		case "user":
			fmt.Fprintf(h, "%s\x00%v\x00", role, m["content"])
			goto done
		}
	}
done:
	// An id with this router's own prefix in it defeats the point of sending a
	// session header at all: the shape must match what the client itself emits.
	return opencodeSessionID(hex.EncodeToString(h.Sum(nil)))
}

// canonicalModel is the model a request gets when it asks for nothing. Clients
// that hardcode a provider-specific id keep working: resolveModel accepts those
// as aliases.
var canonicalModel = "deepseek-v4.1-flash"

// clefRates is Ollama's published clef pricing — input only, $0.24/M. Cached and
// output are not published, so they are charged at the same rate rather than at
// zero, which would make the ledger under-report real spend.
var clefRates = Rates{In: 0.24, Cache: 0.24, Out: 0.24, PeakMult: 1.0}

// defaultProviders is the built-in table, used when no config file is given.
// Every provider here serves deepseek-v4.1-flash under its own id; Ollama
// additionally serves clef, which no other provider has.
func defaultProviders() map[string]*Provider {
	return map[string]*Provider{
		"go": {
			Base: "https://opencode.ai/inference/go/openai/v1",
			Models: map[string]ModelRoute{
				"deepseek-v4.1-flash": {ID: "deepseek-v4.1-flash"},
			},
			KeyEnv:        "OPENCODE_API_KEY",
			SessionHeader: true,
		},
		"ollama": {
			Base: "https://ollama.com/v1",
			Models: map[string]ModelRoute{
				"deepseek-v4.1-flash": {ID: "deepseek-v4.1-flash:1017"},
				// opencode's go provider answers "Model is unavailable" for gemma4,
				// so only ollama carries it. Same canonical name, provider-specific id.
				"gemma4": {ID: "gemma4:31b"},
				"clef":   {ID: "clef:27b", Rates: &clefRates, API: APISystemOne},
			},
			KeyEnv: "OLLAMA_API_KEY",
		},
		"nous": {
			Base: "https://inference-api.nousresearch.com/v1",
			Models: map[string]ModelRoute{
				"deepseek-v4.1-flash": {ID: "deepseek/deepseek-v4.1-flash-1017"},
			},
			// Nous issues a one-hour access token, not an API key: a copy in the
			// environment is dead within the hour, which is why this fallback sat
			// at 401 while the account still had credit. Mint from the refresh token.
			OAuth: &OAuthConfig{
				PortalBase: "https://portal.nousresearch.com",
				ClientID:   "hermes-cli",
				RefreshEnv: "NOUS_REFRESH_TOKEN",
			},
		},
	}
}

// resolveModel maps a requested model name to the canonical name to route on.
// Both canonical names and any provider's upstream id are accepted, so a client
// that asks for deepseek-v4.1-flash:1017 keeps working after the alias table
// exists. An unknown name is refused rather than silently served by a different
// model — a substitution the caller cannot see is worse than an error.
func resolveModel(requested string) (string, bool) {
	if requested == "" || requested == "default" {
		if defaultServed() != "" {
			return canonicalModel, true
		}
		return "", false
	}
	for _, p := range provs {
		if len(p.Models) == 0 {
			if p.Model == requested {
				return requested, true
			}
			continue
		}
		if _, ok := p.Models[requested]; ok {
			return requested, true
		}
	}
	// Claude ids land on the canonical chat model. A Claude Code session sends the
	// model it was configured with plus a small model for its own housekeeping, and an
	// unknown-model 400 there ends the session before any work starts.
	if isClaudeModelID(requested) {
		if target := claudeAliasTarget(); target != "" {
			return target, true
		}
	}
	for _, p := range provs {
		for canonical, route := range p.Models {
			if route.ID == requested {
				return canonical, true
			}
		}
	}
	return "", false
}

// defaultServed returns the canonical model in effect, falling back to any
// served model when the configured default is not offered by any provider.
func defaultServed() string {
	for name := range provs {
		if servesModel(name, canonicalModel) {
			return canonicalModel
		}
	}
	ms := availableModels()
	if len(ms) > 0 {
		return ms[0]
	}
	return ""
}

// modelAPI returns the endpoint family a canonical model speaks, defaulting to
// chat. The answer comes from whichever provider declares it.
func modelAPI(canonical string) string {
	for name := range provs {
		p := provs[name]
		if r, ok := p.Models[canonical]; ok {
			if r.API == "" {
				return APIChat
			}
			return r.API
		}
		if len(p.Models) == 0 && p.Model == canonical {
			return APIChat
		}
	}
	return APIChat
}

// availableModels lists every canonical model any loaded provider can serve.
func availableModels() []string {
	seen := map[string]bool{}
	var out []string
	for _, p := range provs {
		if len(p.Models) == 0 {
			if !seen[p.Model] {
				seen[p.Model] = true
				out = append(out, p.Model)
			}
			continue
		}
		for m := range p.Models {
			if !seen[m] {
				seen[m] = true
				out = append(out, m)
			}
		}
	}
	sort.Strings(out)
	return out
}

// applyEnvOverrides lets a single provider be retargeted without a config file,
// which is what the test suite uses.
func applyEnvOverrides() {
	for name, p := range provs {
		up := strings.ToUpper(strings.ReplaceAll(name, "-", "_"))
		if v := os.Getenv("ROUTER_" + up + "_BASE"); v != "" {
			p.Base = v
		}
		if v := os.Getenv("ROUTER_" + up + "_MODEL"); v != "" {
			p.Model = v
			// With a model table the meaningful override is the upstream id of
			// the canonical model; setting Model alone would be ignored.
			if r, ok := p.Models[canonicalModel]; ok {
				r.ID = v
				p.Models[canonicalModel] = r
			}
		}
		if v := os.Getenv("ROUTER_" + up + "_UA"); v != "" {
			p.UA = v
		}
	}
}

// ── circuit breaker ──────────────────────────────────────────────────────
// A provider returning 403 on every call (unpaid invoice) would otherwise
// cost one wasted round-trip per request.

const (
	breakAfter   = 3
	breakSeconds = 60 * time.Second
)

type breakerEntry struct {
	fails     int
	openUntil time.Time
}

var (
	brMu     sync.Mutex
	breakers = map[string]*breakerEntry{}
)

func breakerOpen(p string) bool {
	brMu.Lock()
	defer brMu.Unlock()
	e, ok := breakers[p]
	return ok && time.Now().Before(e.openUntil)
}

func breakerFail(p string) {
	brMu.Lock()
	defer brMu.Unlock()
	e := breakers[p]
	if e == nil {
		e = &breakerEntry{}
		breakers[p] = e
	}
	e.fails++
	if e.fails >= breakAfter {
		e.openUntil = time.Now().Add(breakSeconds)
	}
}

func breakerOK(p string) {
	brMu.Lock()
	defer brMu.Unlock()
	breakers[p] = &breakerEntry{}
}

func breakerSnapshot() map[string]any {
	brMu.Lock()
	defer brMu.Unlock()
	out := map[string]any{}
	for p, e := range breakers {
		if e.fails == 0 {
			continue
		}
		secs := 0.0
		if d := time.Until(e.openUntil); d > 0 {
			secs = d.Seconds()
		}
		out[p] = map[string]any{"fails": e.fails, "open_for_s": secs}
	}
	return out
}

// ── usage → cost ─────────────────────────────────────────────────────────

type usageBlock struct {
	PromptTokens         int `json:"prompt_tokens"`
	CompletionTokens     int `json:"completion_tokens"`
	PromptCacheHitTokens int `json:"prompt_cache_hit_tokens"`
	PromptTokensDetails  struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

type usageEnvelope struct {
	Usage *usageBlock `json:"usage"`
}

// splitUsage maps a provider usage block to (fresh input, cache reads, output).
func splitUsage(u *usageBlock) (float64, float64, float64) {
	if u == nil {
		return 0, 0, 0
	}
	cache := u.PromptTokensDetails.CachedTokens
	if cache == 0 {
		cache = u.PromptCacheHitTokens
	}
	in := u.PromptTokens - cache
	if in < 0 {
		in = 0
	}
	return float64(in), float64(cache), float64(u.CompletionTokens)
}

var (
	led    *Ledger
	provs  map[string]*Provider
	shadow bool
	// authToken, when non-empty, is required as `Authorization: Bearer <token>`
	// on every endpoint except /health. Empty disables auth entirely.
	authToken string
	// Overridable by the config file.
	cfgListen = "127.0.0.1:8788"
	cfgLedger = "spend.json"
	// nowFn is swappable so tests can pin the clock to a specific UTC hour.
	nowFn = func() time.Time { return time.Now().UTC() }
	// Initialised at package level, not in main(): state that only exists
	// once main() runs is state the tests cannot reach.
	httpc = &http.Client{
		// The wrapper sends the client's full Accept-Encoding and decodes whatever
		// comes back: Go's transport would otherwise only handle gzip, and only
		// when it set the header itself.
		Transport: &decompressTransport{base: &http.Transport{
			ResponseHeaderTimeout: 120 * time.Second,
			MaxIdleConnsPerHost:   8,
		}},
	}
)

func record(provider, model string, u *usageBlock, now time.Time) float64 {
	in, cache, out := splitUsage(u)
	cost := requestCost(provider, model, in, cache, out, isPeak(provider, now))
	led.Add(now, provider, cost)
	return cost
}

// ── handlers ─────────────────────────────────────────────────────────────

// withAuth gates a handler behind the bearer token. /health stays open so a
// supervisor can probe liveness without holding the credential.
func withAuth(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" || authorized(r) {
			h(w, r)
			return
		}
		w.Header().Set("WWW-Authenticate", `Bearer realm="offpeak-router"`)
		writeJSON(w, 401, map[string]any{"error": map[string]string{
			"message": "unauthorized: send Authorization: Bearer <token>",
			"type":    "invalid_request_error",
		}})
	}
}

func authorized(r *http.Request) bool {
	if authToken == "" {
		return true
	}
	const prefix = "Bearer "
	h := r.Header.Get("Authorization")
	if !strings.HasPrefix(h, prefix) {
		return false
	}
	// Constant time: a length leak here is a token oracle.
	return subtle.ConstantTimeCompare(
		[]byte(strings.TrimPrefix(h, prefix)), []byte(authToken)) == 1
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func handleHealth(w http.ResponseWriter, _ *http.Request) {
	keys := map[string]bool{}
	for name, p := range provs {
		keys[name] = providerKeyPresent(name, p)
	}
	writeJSON(w, 200, map[string]any{"status": "ok", "shadow": shadow, "keys": keys})
}

func handleStatus(w http.ResponseWriter, _ *http.Request) {
	now := nowFn()
	spent := led.Spent(now)
	prov := map[string]any{}
	// Iterate whatever is loaded, not a hardcoded list — a config file can name
	// its providers anything.
	for name := range provs {
		r, capped := remaining(name, spent)
		var rem any = nil
		var allow any = nil
		if capped {
			rem = r
			allow = allowance[name]
		}
		prov[name] = map[string]any{
			"peak":          isPeak(name, now),
			"allowance_usd": allow,
			"spent_usd":     spent[name],
			"remaining_usd": rem,
			"hard_cap":      hardCap[name],
			"key_present":   providerKeyPresent(name, provs[name]),
		}
	}
	writeJSON(w, 200, map[string]any{
		"now_utc":       now.Format("2006-01-02 15:04:05"),
		"weekday":       int(now.Weekday()),
		"providers":     prov,
		"order":         candidateOrder(now, spent, ""),
		"models":        availableModels(),
		"default_model": defaultServed(),
		"shadow":        shadow,
		"breakers":      breakerSnapshot(),
		"web": map[string]any{
			"search_calls":   searchCalls.Load(),
			"fetch_calls":    fetchCalls.Load(),
			"search_backend": searchBackendName(),
		},
	})
}

func handleModels(w http.ResponseWriter, _ *http.Request) {
	var data []map[string]any
	for _, m := range availableModels() {
		var by []string
		for name := range provs {
			if servesModel(name, m) {
				by = append(by, name)
			}
		}
		sort.Strings(by)
		data = append(data, map[string]any{
			"id": m, "object": "model", "owned_by": "offpeak-router",
			"providers": by, "api": modelAPI(m),
			// Without these a client has no way to know the model takes images
			// and concludes the router has no vision model at all.
			"capabilities":         capabilitiesFor(m),
			"capability_providers": capabilityProvidersFor(m),
			"input_modalities":     inputModalities(m),
		})
		// A wildcard, advertised deliberately: every Claude id resolves here, so a
		// client cannot be told a fixed list it might outgrow.
		if m == claudeAliasTarget() {
			data[len(data)-1]["alias_patterns"] = []string{"claude-*"}
		}
	}
	writeJSON(w, 200, map[string]any{"object": "list", "data": data})
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSON(w, 400, map[string]any{"error": err.Error()})
		return
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		writeJSON(w, 400, map[string]any{"error": "bad json"})
		return
	}
	stream, _ := body["stream"].(bool)
	now := nowFn()

	requested, _ := body["model"].(string)
	model, ok := resolveModel(requested)
	if !ok {
		writeJSON(w, 400, map[string]any{"error": map[string]any{
			"message":   "unknown model: " + requested,
			"type":      "invalid_request_error",
			"available": availableModels(),
		}})
		return
	}

	order := candidateOrder(now, led.Spent(now), model)

	// A decision model has no chat completions endpoint upstream: sending it
	// here would fail with a 400 on every attempt and blame the provider.
	if api := modelAPI(model); api != APIChat {
		writeJSON(w, 400, map[string]any{"error": map[string]any{
			"message": model + " is a " + api + " model; call /v1/" + api + " instead",
			"type":    "invalid_request_error",
			"model":   model,
			"api":     api,
		}})
		return
	}

	if shadow {
		peak := map[string]bool{}
		for name := range provs {
			peak[name] = isPeak(name, now)
		}
		would := ""
		if len(order) > 0 {
			would = order[0]
		}
		writeJSON(w, 200, map[string]any{
			"shadow": true, "order": order, "would_use": would,
			"peak": peak, "body_model": body["model"], "model": model,
		})
		return
	}

	if len(order) == 0 {
		writeJSON(w, 503, map[string]any{"error": map[string]any{
			"message": "no provider can serve " + model + " right now",
			"type":    "insufficient_capacity",
			"model":   model,
		}})
		return
	}

	var attempts []map[string]string
	anyFailover := false
	for _, name := range order {
		p := provs[name]
		key, keyErr := providerKey(r.Context(), name, p)
		if keyErr != nil {
			attempts = append(attempts, map[string]string{"provider": name, "error": keyErr.Error()})
			continue
		}
		if breakerOpen(name) {
			attempts = append(attempts, map[string]string{"provider": name, "error": "circuit open"})
			continue
		}
		// Headers must be set BEFORE forward() writes them: once WriteHeader
		// has run, Set is silently ignored.
		if anyFailover {
			w.Header().Set("X-Router-Failover", "true")
		}
		if err := forward(w, r, name, p, key, body, stream, now, model); err == nil {
			breakerOK(name)
			return
		} else {
			log.Printf("provider %s failed: %v", name, err)
			breakerFail(name)
			anyFailover = true
			attempts = append(attempts, map[string]string{
				"provider": name, "error": err.Error()})
		}
	}
	writeJSON(w, 502, map[string]any{
		"error": map[string]any{"message": "all providers failed", "attempts": attempts},
	})
}

// systemOneUsage is this API's usage block: plain input and output token
// counts, with no cache accounting to split.
type systemOneUsage struct {
	PromptTokens     int `json:"input_tokens"`
	CompletionTokens int `json:"output_tokens"`
}

// recordTokens bills a non-chat call, whose usage arrives as bare token counts.
func recordTokens(provider, model string, in, out int, now time.Time) float64 {
	cost := requestCost(provider, model, float64(in), 0, float64(out), isPeak(provider, now))
	led.Add(now, provider, cost)
	return cost
}

// handleSystemOne serves decision models: one state plus a schema of typed
// questions, answered in a single pass. There is no streaming, no tools and no
// generation controls on this API, so nothing has to be negotiated.
func handleSystemOne(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSON(w, 400, map[string]any{"error": err.Error()})
		return
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		writeJSON(w, 400, map[string]any{"error": "bad json"})
		return
	}
	now := nowFn()
	requested, _ := body["model"].(string)
	model, ok := resolveModel(requested)
	if !ok {
		writeJSON(w, 400, map[string]any{"error": map[string]any{
			"message":   "unknown model: " + requested,
			"type":      "invalid_request_error",
			"available": availableModels(),
		}})
		return
	}
	if api := modelAPI(model); api != APISystemOne {
		writeJSON(w, 400, map[string]any{"error": map[string]any{
			"message": model + " is a " + api + " model; call /v1/chat/completions instead",
			"type":    "invalid_request_error",
			"model":   model,
			"api":     api,
		}})
		return
	}

	order := candidateOrder(now, led.Spent(now), model)
	if len(order) == 0 {
		writeJSON(w, 503, map[string]any{"error": map[string]any{
			"message": "no provider can serve " + model + " right now",
			"type":    "insufficient_capacity", "model": model,
		}})
		return
	}

	var attempts []map[string]string
	anyFailover := false
	for _, name := range order {
		p := provs[name]
		key, keyErr := providerKey(r.Context(), name, p)
		if keyErr != nil {
			attempts = append(attempts, map[string]string{"provider": name, "error": keyErr.Error()})
			continue
		}
		if breakerOpen(name) {
			attempts = append(attempts, map[string]string{"provider": name, "error": "circuit open"})
			continue
		}
		if anyFailover {
			w.Header().Set("X-Router-Failover", "true")
		}
		if err := forwardSystemOne(w, r, name, p, key, body, now, model); err == nil {
			breakerOK(name)
			return
		} else {
			log.Printf("provider %s failed: %v", name, err)
			breakerFail(name)
			anyFailover = true
			attempts = append(attempts, map[string]string{
				"provider": name, "error": err.Error()})
		}
	}
	writeJSON(w, 502, map[string]any{
		"error": map[string]any{"message": "all providers failed", "attempts": attempts},
	})
}

// forwardSystemOne posts one decision request and bills it from its own usage
// block. It mirrors forward()'s contract: headers are only written once the
// upstream has actually succeeded.
func forwardSystemOne(w http.ResponseWriter, r *http.Request, name string,
	p *Provider, key string, body map[string]any, now time.Time, model string) error {

	payload := make(map[string]any, len(body)+1)
	for k, v := range body {
		payload[k] = v
	}
	payload["model"] = upstreamModel(name, model)

	buf, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("encode body: %w", err)
	}
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		p.Base+"/systemone", bytes.NewReader(buf))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	applyUpstreamHeaders(req, p, r, body)

	resp, err := httpc.Do(req)
	if err != nil {
		return fmt.Errorf("transport: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 400))
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(detail)))
	}
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read body: %w", err)
	}
	var env struct {
		Usage systemOneUsage `json:"usage"`
	}
	_ = json.Unmarshal(raw, &env)
	cost := recordTokens(name, model, env.Usage.PromptTokens, env.Usage.CompletionTokens, now)

	w.Header().Set("X-Router-Provider", name)
	w.Header().Set("X-Router-Model", model)
	w.Header().Set("X-Router-Usd", fmt.Sprintf("%.6f", cost))
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(200)
	_, _ = w.Write(raw)
	return nil
}

// forward writes a complete response to w and returns nil, or returns the
// error without touching w. Callers need the real cause: a bare "upstream
// failed" made a working provider look broken.
func forward(w http.ResponseWriter, r *http.Request, name string, p *Provider,
	key string, body map[string]any, stream bool, now time.Time, model string) error {

	payload := make(map[string]any, len(body)+2)
	for k, v := range body {
		payload[k] = v
	}
	// The client asks for a canonical model; each provider gets the id it
	// actually knows. This line is the whole reason the model table exists.
	payload["model"] = upstreamModel(name, model)
	if stream {
		opts, _ := payload["stream_options"].(map[string]any)
		if opts == nil {
			opts = map[string]any{}
		}
		opts["include_usage"] = true
		payload["stream_options"] = opts
	}

	buf, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("encode body: %w", err)
	}
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		p.Base+"/chat/completions", bytes.NewReader(buf))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	applyUpstreamHeaders(req, p, r, body)

	resp, err := httpc.Do(req)
	if err != nil {
		return fmt.Errorf("transport: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 400))
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode,
			strings.TrimSpace(string(detail)))
	}

	w.Header().Set("X-Router-Provider", name)
	w.Header().Set("X-Router-Model", model)

	if !stream {
		raw, err := io.ReadAll(resp.Body)
		if err != nil {
			return fmt.Errorf("read body: %w", err)
		}
		var env usageEnvelope
		_ = json.Unmarshal(raw, &env)
		cost := record(name, model, env.Usage, now)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Router-Usd", fmt.Sprintf("%.6f", cost))
		w.WriteHeader(200)
		_, _ = w.Write(raw)
		return nil
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(200)
	flusher, _ := w.(http.Flusher)

	var captured *usageBlock
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if data, ok := strings.CutPrefix(line, "data: "); ok && strings.TrimSpace(data) != "[DONE]" {
			var env usageEnvelope
			if json.Unmarshal([]byte(data), &env) == nil && env.Usage != nil {
				captured = env.Usage
			}
		}
		if _, err := io.WriteString(w, line+"\n"); err != nil {
			break
		}
		if flusher != nil {
			flusher.Flush()
		}
	}
	record(name, model, captured, now)
	return nil
}

// newMux is the only place routes are registered, so tests exercise the same
// table the binary serves. A test that builds its own mux silently skips
// whatever middleware main() wraps around these handlers.
func newMux() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", withAuth(handleHealth))
	mux.HandleFunc("/status", withAuth(handleStatus))
	mux.HandleFunc("/v1/models", withAuth(handleModels))
	mux.HandleFunc("/v1/chat/completions", withAuth(handleChat))
	// Anthropic Messages dialect: the clients that speak it cannot use any other
	// route here.
	mux.HandleFunc("/v1/messages", withAuth(handleMessages))
	mux.HandleFunc("/v1/messages/count_tokens", withAuth(handleCountTokens))
	mux.HandleFunc("/v1/systemone", withAuth(handleSystemOne))
	// Web layer: same bearer auth and the same ledger host as the model routes,
	// so a client can point its whole web toolset here instead of a separate key.
	mux.HandleFunc("/v1/search", withAuth(handleSearch))
	mux.HandleFunc("/v1/fetch", withAuth(handleFetch))
	// Ollama-native equivalents of /v1/models: a client that probes for Ollama
	// before trusting a server finds an answer here instead of a 404.
	mux.HandleFunc("/api/tags", withAuth(handleOllamaTags))
	mux.HandleFunc("/api/version", withAuth(handleOllamaVersion))
	// Chat in Ollama's dialect: translated to the OpenAI payload upstream, so
	// routing, failover and billing are the same code path as /v1.
	// Ollama Cloud exposes web search and fetch as /api/web_search and
	// /api/web_fetch in its own body shape. A client written against the ollama
	// SDK calls those exact names, so route names alone are not compatibility.
	mux.HandleFunc("/api/web_search", withAuth(handleOllamaWebSearch))
	mux.HandleFunc("/api/web_fetch", withAuth(handleOllamaWebFetch))
	mux.HandleFunc("/api/chat", withAuth(handleOllamaChat))
	mux.HandleFunc("/api/generate", withAuth(handleOllamaGenerate))
	mux.HandleFunc("/api/show", withAuth(handleOllamaShow))
	mux.HandleFunc("/api/ps", withAuth(handleOllamaPs))
	return mux
}

func main() {
	shadow = os.Getenv("ROUTER_SHADOW") == "1"
	provs = defaultProviders()

	if path := os.Getenv("ROUTER_CONFIG"); path != "" {
		c, err := loadConfig(path)
		if err != nil {
			log.Fatalf("config: %v", err)
		}
		if err := applyConfig(c); err != nil {
			log.Fatalf("config: %v", err)
		}
	}
	applyEnvOverrides()

	if v := os.Getenv("ROUTER_ADDR"); v != "" {
		cfgListen = v
	}
	if v := os.Getenv("ROUTER_DB"); v != "" {
		cfgLedger = v
	}
	led = NewLedger(cfgLedger)

	authToken = os.Getenv("ROUTER_AUTH_TOKEN")
	if authToken != "" {
		log.Printf("auth: bearer token required (set ROUTER_AUTH_TOKEN to change)")
	} else {
		log.Printf("auth: DISABLED — do not expose this listener publicly")
	}

	log.Printf("offpeak-router on %s  shadow=%v  providers=%d", cfgListen, shadow, len(provs))
	srv := &http.Server{
		Addr:              cfgListen,
		Handler:           newMux(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Fatal(srv.ListenAndServe())
}
