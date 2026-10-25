package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

type Provider struct {
	Base   string
	Model  string
	KeyEnv string
}

// defaultProviders is the built-in table, used when no config file is given.
// These three all serve DeepSeek V4.1 Flash; swap them in a config file for
// anything else.
func defaultProviders() map[string]*Provider {
	return map[string]*Provider{
		"go": {
			Base:   "https://opencode.ai/inference/go/openai/v1",
			Model:  "deepseek-v4.1-flash",
			KeyEnv: "OPENCODE_API_KEY",
		},
		"ollama": {
			Base:   "https://ollama.com/v1",
			Model:  "deepseek-v4.1-flash:1017",
			KeyEnv: "OLLAMA_API_KEY",
		},
		"nous": {
			Base:   "https://inference-api.nousresearch.com/v1",
			Model:  "deepseek/deepseek-v4.1-flash-1017",
			KeyEnv: "NOUS_API_KEY",
		},
	}
}

// applyEnvOverrides lets a single provider be retargeted without a config file,
// which is what the test suite uses.
func applyEnvOverrides() {
	for name, p := range provs {
		up := strings.ToUpper(name)
		if v := os.Getenv("ROUTER_" + up + "_BASE"); v != "" {
			p.Base = v
		}
		if v := os.Getenv("ROUTER_" + up + "_MODEL"); v != "" {
			p.Model = v
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
	// Overridable by the config file.
	cfgListen = "127.0.0.1:8788"
	cfgLedger = "spend.json"
	// nowFn is swappable so tests can pin the clock to a specific UTC hour.
	nowFn = func() time.Time { return time.Now().UTC() }
	// Initialised at package level, not in main(): state that only exists
	// once main() runs is state the tests cannot reach.
	httpc = &http.Client{
		Transport: &http.Transport{
			ResponseHeaderTimeout: 120 * time.Second,
			MaxIdleConnsPerHost:   8,
		},
	}
)

func record(provider string, u *usageBlock, now time.Time) float64 {
	in, cache, out := splitUsage(u)
	cost := requestCost(provider, in, cache, out, isPeak(provider, now))
	led.Add(now, provider, cost)
	return cost
}

// ── handlers ─────────────────────────────────────────────────────────────

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func handleHealth(w http.ResponseWriter, _ *http.Request) {
	keys := map[string]bool{}
	for name, p := range provs {
		keys[name] = os.Getenv(p.KeyEnv) != ""
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
			"key_present":   os.Getenv(provs[name].KeyEnv) != "",
		}
	}
	writeJSON(w, 200, map[string]any{
		"now_utc":   now.Format("2006-01-02 15:04:05"),
		"weekday":   int(now.Weekday()),
		"providers": prov,
		"order":     candidateOrder(now, spent),
		"shadow":    shadow,
		"breakers":  breakerSnapshot(),
	})
}

func handleModels(w http.ResponseWriter, _ *http.Request) {
	var data []map[string]any
	for name, p := range provs {
		data = append(data, map[string]any{
			"id": p.Model, "object": "model", "owned_by": name})
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
	order := candidateOrder(now, led.Spent(now))

	if shadow {
		peak := map[string]bool{}
		for name := range provs {
			peak[name] = isPeak(name, now)
		}
		writeJSON(w, 200, map[string]any{
			"shadow": true, "order": order, "would_use": order[0],
			"peak": peak, "body_model": body["model"],
		})
		return
	}

	var attempts []map[string]string
	anyFailover := false
	for _, name := range order {
		p := provs[name]
		key := os.Getenv(p.KeyEnv)
		if key == "" {
			attempts = append(attempts, map[string]string{"provider": name, "error": "no api key"})
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
		if forward(w, r, name, p, key, body, stream, now) {
			breakerOK(name)
			return
		}
		breakerFail(name)
		anyFailover = true
		attempts = append(attempts, map[string]string{"provider": name, "error": "upstream failed"})
	}
	writeJSON(w, 502, map[string]any{
		"error": map[string]any{"message": "all providers failed", "attempts": attempts},
	})
}

// forward returns true once it has written a complete response to w.
func forward(w http.ResponseWriter, r *http.Request, name string, p *Provider,
	key string, body map[string]any, stream bool, now time.Time) bool {

	payload := make(map[string]any, len(body)+2)
	for k, v := range body {
		payload[k] = v
	}
	payload["model"] = p.Model
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
		return false
	}
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		p.Base+"/chat/completions", bytes.NewReader(buf))
	if err != nil {
		return false
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)

	resp, err := httpc.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		_, _ = io.Copy(io.Discard, resp.Body)
		return false
	}

	w.Header().Set("X-Router-Provider", name)

	if !stream {
		raw, err := io.ReadAll(resp.Body)
		if err != nil {
			return false
		}
		var env usageEnvelope
		_ = json.Unmarshal(raw, &env)
		cost := record(name, env.Usage, now)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Router-Usd", fmt.Sprintf("%.6f", cost))
		w.WriteHeader(200)
		_, _ = w.Write(raw)
		return true
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
	record(name, captured, now)
	return true
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

	mux := http.NewServeMux()
	mux.HandleFunc("/health", handleHealth)
	mux.HandleFunc("/status", handleStatus)
	mux.HandleFunc("/v1/models", handleModels)
	mux.HandleFunc("/v1/chat/completions", handleChat)

	log.Printf("offpeak-router on %s  shadow=%v  providers=%d", cfgListen, shadow, len(provs))
	srv := &http.Server{
		Addr:              cfgListen,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Fatal(srv.ListenAndServe())
}
