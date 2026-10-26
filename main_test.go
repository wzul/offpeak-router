package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const usageJSON = `{"prompt_tokens":1000,"completion_tokens":100,` +
	`"prompt_tokens_details":{"cached_tokens":800}}`

type upstream struct {
	mu       sync.Mutex
	hits     []string        // provider path segment, in order
	fail     map[string]bool // path segment -> always 500
	lastUA   string
	sessions []string // x-opencode-session seen, in order
}

func (u *upstream) record(p string) bool {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.hits = append(u.hits, p)
	return u.fail[p]
}

func (u *upstream) reset() {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.hits = nil
}

func (u *upstream) calls() []string {
	u.mu.Lock()
	defer u.mu.Unlock()
	out := make([]string, len(u.hits))
	copy(out, u.hits)
	return out
}

func newUpstream(t *testing.T) (*upstream, *httptest.Server) {
	t.Helper()
	u := &upstream{fail: map[string]bool{}}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := strings.TrimPrefix(r.URL.Path, "/")
		p = strings.SplitN(p, "/", 2)[0]
		if u.record(p) {
			w.WriteHeader(500)
			_, _ = io.WriteString(w, `{"error":"boom"}`)
			return
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		u.mu.Lock()
		u.lastUA = r.Header.Get("User-Agent")
		u.sessions = append(u.sessions, r.Header.Get("x-opencode-session"))
		u.mu.Unlock()
		// Decision models answer on a different shape entirely.
		if strings.HasSuffix(r.URL.Path, "/systemone") {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{"model":%q,"answers":{"ok":{"type":"noul","noul":0.99}},`+
				`"usage":{"input_tokens":1000,"output_tokens":0}}`, body["model"])
			return
		}
		// echo back which model the router asked for, so tests can assert rewrite
		if stream, _ := body["stream"].(bool); stream {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(200)
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
			fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":%s}\n\n", usageJSON)
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"choices":[{"message":{"content":"hi"}}],`+
			`"echo_model":%q,"usage":%s}`, body["model"], usageJSON)
	}))
	t.Cleanup(s.Close)
	return u, s
}

func setup(t *testing.T, up *httptest.Server) *httptest.Server {
	t.Helper()
	t.Setenv("OPENCODE_API_KEY", "k-go")
	t.Setenv("OLLAMA_API_KEY", "k-ollama")
	t.Setenv("NOUS_API_KEY", "k-nous")

	// Leaving provs replaced would leak into later tests in the same package —
	// the policy tests then run against a three-provider world and never see
	// a provider this test never stubbed.
	origProvs := provs
	t.Cleanup(func() { provs = origProvs })

	provs = map[string]*Provider{
		// All three serve one canonical model under their own upstream id, which
		// is exactly the shape production has: one name for the client, three
		// spellings for the providers.
		"go": {Base: up.URL + "/go/v1", KeyEnv: "OPENCODE_API_KEY",
			Models: map[string]ModelRoute{"mock-model": {ID: "mock-go"}}},
		"ollama": {Base: up.URL + "/ollama/v1", KeyEnv: "OLLAMA_API_KEY",
			Models: map[string]ModelRoute{"mock-model": {ID: "mock-ollama"}}},
		"nous": {Base: up.URL + "/nous/v1", KeyEnv: "NOUS_API_KEY",
			Models: map[string]ModelRoute{"mock-model": {ID: "mock-nous"}}},
	}
	led = NewLedger(filepath.Join(t.TempDir(), "spend.json"))
	breakers = map[string]*breakerEntry{}
	shadow = false
	authToken = ""
	nowFn = func() time.Time { return mon(0) }

	s := httptest.NewServer(newMux())
	t.Cleanup(s.Close)
	return s
}

func ask(t *testing.T, router *httptest.Server, stream bool) (*http.Response, []byte) {
	t.Helper()
	payload := fmt.Sprintf(`{"model":"mock-model","stream":%v,`+
		`"messages":[{"role":"user","content":"hi"}]}`, stream)
	resp, err := http.Post(router.URL+"/v1/chat/completions",
		"application/json", strings.NewReader(payload))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	b, _ := io.ReadAll(resp.Body)
	return resp, b
}

func atHour(h int) { nowFn = func() time.Time { return mon(h) } }

func TestRoutingByHour(t *testing.T) {
	up, ups := newUpstream(t)
	router := setup(t, ups)

	cases := []struct {
		hour int
		want string
	}{
		{0, "go"}, {2, "ollama"}, {8, "ollama"}, {11, "go"}, {13, "go"}, {17, "go"},
	}
	for _, c := range cases {
		up.reset()
		atHour(c.hour)
		resp, _ := ask(t, router, false)
		if got := resp.Header.Get("X-Router-Provider"); got != c.want {
			t.Errorf("%02d:00 routed to %q, want %q", c.hour, got, c.want)
		}
		if hits := up.calls(); len(hits) != 1 || hits[0] != c.want {
			t.Errorf("%02d:00 upstream hits = %v, want [%s]", c.hour, hits, c.want)
		}
	}
}

func TestModelRewritten(t *testing.T) {
	_, ups := newUpstream(t)
	router := setup(t, ups)
	atHour(13) // -> go
	_, body := ask(t, router, false)
	if !strings.Contains(string(body), `"echo_model":"mock-go"`) {
		t.Errorf("model not rewritten to provider model: %s", body)
	}
}

func TestCostRecorded(t *testing.T) {
	_, ups := newUpstream(t)
	router := setup(t, ups)
	atHour(13) // go off-peak
	resp, _ := ask(t, router, false)

	want := (200*0.15 + 800*0.003 + 100*0.60) / 1e6
	if got := resp.Header.Get("X-Router-Usd"); got != fmt.Sprintf("%.6f", want) {
		t.Errorf("X-Router-Usd = %q, want %.6f", got, want)
	}
	spent := led.Spent(mon(13))
	if spent["go"] <= 0 {
		t.Error("ledger did not record go spend")
	}
	if spent["ollama"] != 0 {
		t.Error("ollama should be untouched")
	}
}

func TestFailover(t *testing.T) {
	up, ups := newUpstream(t)
	router := setup(t, ups)
	up.fail["go"] = true
	atHour(13)

	resp, _ := ask(t, router, false)
	if got := resp.Header.Get("X-Router-Provider"); got != "ollama" {
		t.Errorf("failover landed on %q, want ollama", got)
	}
	if resp.Header.Get("X-Router-Failover") != "true" {
		t.Error("failover not flagged")
	}
	if hits := up.calls(); len(hits) != 2 || hits[0] != "go" || hits[1] != "ollama" {
		t.Errorf("hits = %v, want [go ollama]", hits)
	}
}

func TestBudgetExhaustion(t *testing.T) {
	up, ups := newUpstream(t)
	router := setup(t, ups)

	led.Add(mon(0), "go", 80)
	atHour(0)
	resp, _ := ask(t, router, false)
	if got := resp.Header.Get("X-Router-Provider"); got != "ollama" {
		t.Errorf("go exhausted at 00:00 -> %q, want ollama", got)
	}

	led.Add(mon(0), "ollama", 60)
	atHour(2) // ollama off-peak, so its top-up beats nous
	resp, _ = ask(t, router, false)
	if got := resp.Header.Get("X-Router-Provider"); got != "ollama" {
		t.Errorf("both exhausted at 02:00 -> %q, want ollama topup", got)
	}

	atHour(13) // ollama peak, so nous beats ollama's peak rate
	resp, _ = ask(t, router, false)
	if got := resp.Header.Get("X-Router-Provider"); got != "nous" {
		t.Errorf("both exhausted at 13:00 -> %q, want nous", got)
	}
	_ = up
}

func TestCircuitBreaker(t *testing.T) {
	up, ups := newUpstream(t)
	router := setup(t, ups)
	up.fail["go"] = true
	atHour(13)

	for i := 0; i < 3; i++ {
		ask(t, router, false)
	}
	if !breakerOpen("go") {
		t.Fatal("breaker should be open after 3 failures")
	}

	up.reset()
	resp, _ := ask(t, router, false)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d with breaker open", resp.StatusCode)
	}
	if got := resp.Header.Get("X-Router-Provider"); got != "ollama" {
		t.Errorf("breaker open -> %q, want ollama", got)
	}
	if hits := up.calls(); len(hits) != 1 || hits[0] != "ollama" {
		t.Errorf("wasted a call to the dead provider: %v", hits)
	}

	// a success closes it again
	up.reset()
	breakers = map[string]*breakerEntry{}
	up.fail["go"] = false
	resp, _ = ask(t, router, false)
	if got := resp.Header.Get("X-Router-Provider"); got != "go" {
		t.Errorf("after recovery -> %q, want go", got)
	}
	if breakerOpen("go") {
		t.Error("breaker should be closed on success")
	}
}

func TestMissingKeySkipped(t *testing.T) {
	up, ups := newUpstream(t)
	router := setup(t, ups)
	t.Setenv("OPENCODE_API_KEY", "")
	atHour(13)

	resp, _ := ask(t, router, false)
	if got := resp.Header.Get("X-Router-Provider"); got != "ollama" {
		t.Errorf("keyless go -> %q, want ollama", got)
	}
	if hits := up.calls(); len(hits) != 1 || hits[0] != "ollama" {
		t.Errorf("attempted the keyless provider: %v", hits)
	}
}

func TestShadowMakesNoCall(t *testing.T) {
	up, ups := newUpstream(t)
	router := setup(t, ups)
	shadow = true
	defer func() { shadow = false }()
	atHour(2)

	_, body := ask(t, router, false)
	var out map[string]any
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("shadow body: %v", err)
	}
	if out["would_use"] != "ollama" {
		t.Errorf("shadow would_use = %v, want ollama", out["would_use"])
	}
	if hits := up.calls(); len(hits) != 0 {
		t.Errorf("shadow made %d upstream calls", len(hits))
	}
}

func TestStreamingRelaysAndBills(t *testing.T) {
	_, ups := newUpstream(t)
	router := setup(t, ups)
	atHour(13)

	resp, body := ask(t, router, true)
	if resp.StatusCode != 200 {
		t.Fatalf("stream status %d", resp.StatusCode)
	}
	text := string(body)
	if !strings.Contains(text, "data:") {
		t.Error("no SSE frames relayed")
	}
	if !strings.Contains(text, "[DONE]") {
		t.Error("[DONE] not relayed")
	}
	if spent := led.Spent(mon(13)); spent["go"] <= 0 {
		t.Error("streaming usage not billed")
	}
}

func TestAllDownReturns502(t *testing.T) {
	up, ups := newUpstream(t)
	router := setup(t, ups)
	for _, p := range []string{"go", "ollama", "nous"} {
		up.fail[p] = true
	}
	atHour(0)

	resp, body := ask(t, router, false)
	if resp.StatusCode != 502 {
		t.Errorf("status = %d, want 502", resp.StatusCode)
	}
	var out struct {
		Error struct{ Attempts []map[string]string } `json:"error"`
	}
	_ = json.Unmarshal(body, &out)
	if len(out.Error.Attempts) != 3 {
		t.Errorf("attempts = %d, want 3 (%s)", len(out.Error.Attempts), body)
	}
}

// A request with no User-Agent is refused by Cloudflare in front of at least
// one provider, so the router must always identify itself. This is the
// regression test for a real 403 seen in production.
func TestUserAgentSent(t *testing.T) {
	up, ups := newUpstream(t)
	router := setup(t, ups)
	atHour(0)

	ask(t, router, false)
	up.mu.Lock()
	got := up.lastUA
	up.mu.Unlock()

	if got == "" {
		t.Fatal("no User-Agent sent — upstreams behind Cloudflare answer 403")
	}
	// Naming this router in the User-Agent tells a third-party provider exactly
	// which gateway is calling. The identity sent is the client the provider
	// already expects, so the request is indistinguishable from that client.
	if got != OpenCodeUserAgent {
		t.Errorf("User-Agent = %q, want %q", got, OpenCodeUserAgent)
	}
	if strings.Contains(got, "Go-http-client") {
		t.Error("Go's default User-Agent is the one Cloudflare blocks")
	}
}

// OpenCode Go rejects a request with no x-opencode-session ("MissingSessionID")
// and uses the id to optimise routing and prompt caching. It has to be present,
// stable for a conversation, and different across conversations.
func TestSessionHeader(t *testing.T) {
	up, ups := newUpstream(t)
	router := setup(t, ups)

	// go must send it, and test routing reaches go at 13:00 (go off-peak).
	provs["go"].SessionHeader = true
	atHour(13)

	send := func(content string) {
		body := fmt.Sprintf(`{"model":"mock-model","messages":[{"role":"system","content":"sys"},`+
			`{"role":"user","content":%q}]}`, content)
		resp, err := http.Post(router.URL+"/v1/chat/completions",
			"application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
	}

	send("hello")
	send("hello")     // same conversation prefix -> same id
	send("different") // different conversation -> different id

	up.mu.Lock()
	got := append([]string(nil), up.sessions...)
	up.mu.Unlock()

	if len(got) != 3 {
		t.Fatalf("saw %d requests, want 3", len(got))
	}
	for i, s := range got {
		if s == "" {
			t.Fatalf("request %d carried no x-opencode-session", i)
		}
	}
	if got[0] != got[1] {
		t.Errorf("session id not stable for the same conversation: %q vs %q", got[0], got[1])
	}
	if got[0] == got[2] {
		t.Errorf("session id identical for a different conversation: %q", got[0])
	}
}

// A client that already tracks sessions must have its id preserved.
func TestSessionHeaderFromClientPreserved(t *testing.T) {
	up, ups := newUpstream(t)
	router := setup(t, ups)
	provs["go"].SessionHeader = true
	atHour(13)

	body := `{"model":"mock-model","messages":[{"role":"user","content":"hi"}]}`
	req, err := http.NewRequest(http.MethodPost, router.URL+"/v1/chat/completions",
		strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-opencode-session", "client-session-abc")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	up.mu.Lock()
	got := up.sessions[len(up.sessions)-1]
	up.mu.Unlock()
	if got != "client-session-abc" {
		t.Errorf("client session id not preserved: %q", got)
	}
}

// Providers that do not ask for the header must not receive it.
func TestSessionHeaderOnlyWhenConfigured(t *testing.T) {
	up, ups := newUpstream(t)
	router := setup(t, ups)
	provs["go"].SessionHeader = false
	provs["ollama"].SessionHeader = false
	atHour(13)

	body := `{"model":"mock-model","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(router.URL+"/v1/chat/completions",
		"application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	up.mu.Lock()
	got := up.sessions[len(up.sessions)-1]
	up.mu.Unlock()
	if got != "" {
		t.Errorf("unconfigured provider got x-opencode-session %q", got)
	}
}

// Auth must reject a missing or wrong token and let the right one through,
// while leaving /health open for a supervisor.
func TestAuthRequired(t *testing.T) {
	_, ups := newUpstream(t)
	router := setup(t, ups)
	authToken = "s3cret-token"
	defer func() { authToken = "" }()
	atHour(0)

	post := func(auth string) int {
		body := `{"model":"mock-model","messages":[{"role":"user","content":"hi"}]}`
		req, err := http.NewRequest(http.MethodPost,
			router.URL+"/v1/chat/completions", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		if auth != "" {
			req.Header.Set("Authorization", auth)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		_, _ = io.Copy(io.Discard, resp.Body)
		return resp.StatusCode
	}

	if got := post(""); got != 401 {
		t.Errorf("no token: status %d, want 401", got)
	}
	if got := post("Bearer wrong"); got != 401 {
		t.Errorf("wrong token: status %d, want 401", got)
	}
	if got := post("Basic abc"); got != 401 {
		t.Errorf("wrong scheme: status %d, want 401", got)
	}
	if got := post("Bearer s3cret-token"); got != 200 {
		t.Errorf("right token: status %d, want 200", got)
	}

	// /health must stay open, /status must not.
	resp, err := http.Get(router.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Errorf("/health status %d, want 200 without auth", resp.StatusCode)
	}
	resp, err = http.Get(router.URL + "/status")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 401 {
		t.Errorf("/status status %d, want 401 without auth", resp.StatusCode)
	}
}

// /status must describe whatever providers are loaded, not a built-in list.
func TestStatusUsesLoadedProviders(t *testing.T) {
	_, ups := newUpstream(t)
	router := setup(t, ups)

	resp, err := http.Get(router.URL + "/status")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var st struct {
		Providers map[string]map[string]any `json:"providers"`
		Order     []string                  `json:"order"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&st); err != nil {
		t.Fatal(err)
	}
	if len(st.Providers) != 3 {
		t.Errorf("status listed %d providers, want 3", len(st.Providers))
	}
	for _, want := range []string{"go", "ollama", "nous"} {
		if _, ok := st.Providers[want]; !ok {
			t.Errorf("status missing loaded provider %q", want)
		}
	}
	if len(st.Order) != 3 {
		t.Errorf("order = %v, want 3 entries", st.Order)
	}
}

func TestStaticEndpoints(t *testing.T) {
	_, ups := newUpstream(t)
	router := setup(t, ups)

	resp, err := http.Get(router.URL + "/health")
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("health: %v %d", err, resp.StatusCode)
	}
	resp.Body.Close()

	resp, err = http.Get(router.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	var models struct{ Data []map[string]any }
	_ = json.NewDecoder(resp.Body).Decode(&models)
	resp.Body.Close()
	// One canonical entry, not one per provider: the client asks for a name, not
	// for an upstream spelling. The provider list is what varies.
	if len(models.Data) != 1 {
		t.Fatalf("models = %d, want 1 canonical entry", len(models.Data))
	}
	if id, _ := models.Data[0]["id"].(string); id != "mock-model" {
		t.Errorf("model id = %q, want mock-model", id)
	}
	by, _ := models.Data[0]["providers"].([]any)
	if len(by) != 3 {
		t.Errorf("providers = %v, want all 3", models.Data[0]["providers"])
	}

	resp, err = http.Get(router.URL + "/status")
	if err != nil {
		t.Fatal(err)
	}
	var st map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&st)
	resp.Body.Close()
	if _, ok := st["providers"]; !ok {
		t.Error("status missing providers")
	}
}
