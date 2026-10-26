package main

// The router presents itself as the client the provider expects. Two things made
// that false before: a User-Agent naming this router, and a session id prefixed
// with its name. These tests pin the whole identity surface, not just one header,
// because a single leftover header is enough to identify the proxy.

import (
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"
)

// identityUpstream records every header set of every request it receives.
func identityUpstream(t *testing.T) (*httptest.Server, func() []http.Header) {
	t.Helper()
	var mu sync.Mutex
	var seen []http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seen = append(seen, r.Header.Clone())
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
	}))
	t.Cleanup(srv.Close)
	return srv, func() []http.Header {
		mu.Lock()
		defer mu.Unlock()
		return append([]http.Header(nil), seen...)
	}
}

func identityRouter(t *testing.T, up *httptest.Server) *httptest.Server {
	t.Helper()
	orig := provs
	t.Cleanup(func() { provs = orig })
	provs = map[string]*Provider{
		"go": {Base: up.URL + "/go/v1", KeyEnv: "OPENCODE_API_KEY",
			SessionHeader: true,
			Models:        map[string]ModelRoute{"mock-model": {ID: "mock-go"}}},
	}
	t.Setenv("OPENCODE_API_KEY", "k")
	led = NewLedger(t.TempDir() + "/spend.json")
	breakers = map[string]*breakerEntry{}
	shadow = false
	authToken = ""
	nowFn = func() time.Time { return mon(13) }
	srv := httptest.NewServer(newMux())
	t.Cleanup(srv.Close)
	return srv
}

// No header may carry this router's name: that is the whole point of the disguise.
func TestNoHeaderNamesTheRouter(t *testing.T) {
	up, headers := identityUpstream(t)
	router := identityRouter(t, up)

	postAnthropic(t, router.URL, "/v1/messages",
		`{"model":"mock-model","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`)

	got := headers()
	if len(got) == 0 {
		t.Fatal("no upstream request was made")
	}
	for _, h := range got {
		for name, values := range h {
			for _, v := range values {
				low := strings.ToLower(v)
				for _, tell := range []string{"offpeak", "router", "hermes"} {
					if strings.Contains(low, tell) {
						t.Errorf("header %s leaks %q: %q", name, tell, v)
					}
				}
			}
		}
	}
}

// The id must have the shape the CLI emits, or the header that was added to look
// like opencode ends up looking like a hand-rolled proxy.
func TestSessionIdentityShape(t *testing.T) {
	up, headers := identityUpstream(t)
	router := identityRouter(t, up)

	body := `{"model":"mock-model","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`
	postAnthropic(t, router.URL, "/v1/messages", body)

	got := headers()
	if len(got) == 0 {
		t.Fatal("no upstream request was made")
	}
	h := got[0]

	idPattern := regexp.MustCompile(`^ses_[0-9A-Za-z]{24}$`)
	session := h.Get("x-opencode-session")
	if !idPattern.MatchString(session) {
		t.Errorf("x-opencode-session = %q, want ses_ plus 24 mixed-case characters", session)
	}
	// The CLI sends the id under four names; a provider or pipeline may key on any.
	for _, name := range []string{"x-opencode-session-id", "x-session-affinity", "x-session-id"} {
		if got := h.Get(name); got != session {
			t.Errorf("%s = %q, want the same id as x-opencode-session (%q)", name, got, session)
		}
	}
	if got := h.Get("x-opencode-client"); got != "cli" {
		t.Errorf("x-opencode-client = %q, want cli", got)
	}
	if got := h.Get("x-opencode-project"); !regexp.MustCompile(`^[0-9a-f]{32}$`).MatchString(got) {
		t.Errorf("x-opencode-project = %q, want a 32-character project hash", got)
	}
	if got := h.Get("traceparent"); !regexp.MustCompile(`^00-[0-9a-f]{32}-[0-9a-f]{16}-01$`).MatchString(got) {
		t.Errorf("traceparent = %q, want W3C trace context", got)
	}
	if got := h.Get("b3"); !regexp.MustCompile(`^[0-9a-f]{32}-[0-9a-f]{16}-1-[0-9a-f]{16}$`).MatchString(got) {
		t.Errorf("b3 = %q, want the Zipkin multi-header form the CLI sends", got)
	}
	if got := h.Get("User-Agent"); got != OpenCodeUserAgent {
		t.Errorf("User-Agent = %q, want %q", got, OpenCodeUserAgent)
	}
}

// Session affinity is the reason the id is derived rather than random: prompt
// caching keys on it, so a second turn of the same conversation must carry the
// same value, and a different conversation must not.
func TestSessionIdentityIsStablePerConversation(t *testing.T) {
	up, headers := identityUpstream(t)
	router := identityRouter(t, up)

	first := `{"model":"mock-model","max_tokens":16,"messages":[
		{"role":"system","content":"be brief"},{"role":"user","content":"first question"}]}`
	again := `{"model":"mock-model","max_tokens":16,"messages":[
		{"role":"system","content":"be brief"},{"role":"user","content":"first question"},
		{"role":"assistant","content":"answer"},{"role":"user","content":"follow up"}]}`
	other := `{"model":"mock-model","max_tokens":16,"messages":[
		{"role":"user","content":"a different conversation"}]}`

	for _, body := range []string{first, again, other} {
		postAnthropic(t, router.URL, "/v1/messages", body)
	}
	got := headers()
	if len(got) != 3 {
		t.Fatalf("upstream calls = %d, want 3", len(got))
	}
	a := got[0].Get("x-opencode-session")
	b := got[1].Get("x-opencode-session")
	c := got[2].Get("x-opencode-session")
	if a == "" || a != b {
		t.Errorf("the same conversation got %q then %q; prompt caching keys on this id", a, b)
	}
	if a == c {
		t.Errorf("two different conversations share the id %q", a)
	}
}

// A client that already sends its own session id must keep it: that is real
// affinity, not a guess.
func TestClientSessionHeaderWins(t *testing.T) {
	up, headers := identityUpstream(t)
	router := identityRouter(t, up)

	req, err := http.NewRequest("POST", router.URL+"/v1/messages",
		strings.NewReader(`{"model":"mock-model","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-opencode-session", "ses_clientOwnedId0000000000")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	got := headers()
	if len(got) == 0 {
		t.Fatal("no upstream request was made")
	}
	if v := got[0].Get("x-opencode-session"); v != "ses_clientOwnedId0000000000" {
		t.Errorf("session id = %q, want the client's own", v)
	}
}

// ROUTER_UA exists so a client version bump does not need a rebuild.
func TestUserAgentOverride(t *testing.T) {
	up, headers := identityUpstream(t)
	router := identityRouter(t, up)
	t.Setenv("ROUTER_UA", "opencode/latest/2.2.0/cli")

	postAnthropic(t, router.URL, "/v1/messages",
		`{"model":"mock-model","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`)
	got := headers()
	if len(got) == 0 || got[0].Get("User-Agent") != "opencode/latest/2.2.0/cli" {
		t.Errorf("User-Agent = %q, want the ROUTER_UA override", got[0].Get("User-Agent"))
	}
}

// Every route family must send the same identity: a fix in one handler is a fix
// the other two never get.
func TestIdentityOnEveryDialect(t *testing.T) {
	up, headers := identityUpstream(t)
	router := identityRouter(t, up)

	bodies := []struct{ path, body string }{
		{"/v1/chat/completions", `{"model":"mock-model","messages":[{"role":"user","content":"hi"}]}`},
		{"/api/chat", `{"model":"mock-model","stream":false,"messages":[{"role":"user","content":"hi"}]}`},
		{"/v1/messages", `{"model":"mock-model","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`},
	}
	for _, b := range bodies {
		resp, raw := postAnthropic(t, router.URL, b.path, b.body)
		if resp.StatusCode != 200 {
			t.Fatalf("%s = %d: %s", b.path, resp.StatusCode, raw)
		}
	}
	got := headers()
	if len(got) != 3 {
		t.Fatalf("upstream calls = %d, want one per dialect", len(got))
	}
	for i, h := range got {
		if h.Get("User-Agent") != OpenCodeUserAgent {
			t.Errorf("%s: User-Agent = %q", bodies[i].path, h.Get("User-Agent"))
		}
		if !strings.HasPrefix(h.Get("x-opencode-session"), "ses_") {
			t.Errorf("%s: session = %q, want the ses_ shape", bodies[i].path, h.Get("x-opencode-session"))
		}
	}
}
