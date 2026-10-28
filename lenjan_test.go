package main

// The spare account is only useful if it is genuinely first, and the enrolment endpoint is
// only safe if it is genuinely one-shot. Both are asserted here rather than trusted.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// The pin must beat every peak and price rule, in both directions of the day.
func TestFirstProviderRunsAheadOfPeakPolicy(t *testing.T) {
	prev := provs
	t.Cleanup(func() { provs = prev })
	provs = defaultProviders()

	for _, hour := range []int{0, 2, 8, 11, 13, 17, 23} {
		order := candidateOrder(mon(hour), zero(), "deepseek-v4.1-flash")
		if len(order) == 0 || order[0] != "ollama-lenjan" {
			t.Errorf("%02d:00 order = %v, want ollama-lenjan first", hour, order)
		}
	}
	// clef rides the same subscription, so it is pinned too - including at peak.
	if order := candidateOrder(mon(13), zero(), "clef"); len(order) == 0 || order[0] != "ollama-lenjan" {
		t.Errorf("clef order = %v, want ollama-lenjan first", order)
	}

	// The pin is not global: a model the spare cannot serve must not put it in front. This
	// uses a fixture rather than the live registry, because every model the ollama accounts
	// serve is served by both of them.
	prevProvs := provs
	t.Cleanup(func() { provs = prevProvs })
	provs = map[string]*Provider{
		"spare": {Base: "http://spare", KeyEnv: "SPARE_KEY", First: true,
			Models: map[string]ModelRoute{"alpha": {ID: "a"}}},
		"other": {Base: "http://other", KeyEnv: "OTHER_KEY",
			Models: map[string]ModelRoute{"beta": {ID: "b"}}},
	}
	for _, name := range candidateOrder(mon(13), zero(), "beta") {
		if name == "spare" {
			t.Error("the pin put a provider in front for a model it does not serve")
		}
	}
	if order := candidateOrder(mon(13), zero(), "alpha"); len(order) == 0 || order[0] != "spare" {
		t.Errorf("alpha order = %v, want spare first", order)
	}
}

// A pinned provider that has failed repeatedly must drop out, or "try this first" becomes
// "fail every request once".
func TestPinnedProviderDropsOutWhenBreakerOpens(t *testing.T) {
	prev := provs
	t.Cleanup(func() { provs = prev })
	provs = defaultProviders()
	oldBreakers := breakers
	t.Cleanup(func() { breakers = oldBreakers })
	breakers = map[string]*breakerEntry{}

	if order := candidateOrder(mon(13), zero(), "deepseek-v4.1-flash"); order[0] != "ollama-lenjan" {
		t.Fatalf("precondition: order = %v", order)
	}
	for i := 0; i < breakAfter; i++ {
		breakerFail("ollama-lenjan")
	}
	if !breakerOpen("ollama-lenjan") {
		t.Fatal("breaker should be open after repeated failures")
	}
	if breakerOpen("ollama") {
		t.Fatal("another provider's breaker must not open")
	}
}

// Ordering cannot skip an open breaker by itself - the dispatcher does - so the claim is
// asserted where it is enforced: with the spare's breaker open, a request must never reach
// it, and must still be answered by the chain.
func TestOpenBreakerSkipsThePinnedProvider(t *testing.T) {
	up := &twoUpstreams{hits: map[string]int{}}
	srv := up.server(t)
	t.Cleanup(srv.Close)

	prev := provs
	t.Cleanup(func() { provs = prev })
	provs = map[string]*Provider{
		"ollama-lenjan": {Base: srv.URL + "/spare", KeyEnv: "SPARE_KEY", First: true,
			Models: map[string]ModelRoute{"m": {ID: "spare-id"}}},
		"ollama": {Base: srv.URL + "/main", KeyEnv: "MAIN_KEY",
			Models: map[string]ModelRoute{"m": {ID: "main-id"}}},
	}
	t.Setenv("SPARE_KEY", "k1")
	t.Setenv("MAIN_KEY", "k2")
	led = NewLedger(t.TempDir() + "/spend.json")
	breakers = map[string]*breakerEntry{}
	shadow = false
	authToken = ""
	nowFn = func() time.Time { return mon(13) }

	for i := 0; i < breakAfter; i++ {
		breakerFail("ollama-lenjan")
	}

	router := httptest.NewServer(newMux())
	t.Cleanup(router.Close)

	resp, raw := postAnthropic(t, router.URL, "/v1/chat/completions",
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d: %s", resp.StatusCode, raw)
	}
	up.mu.Lock()
	defer up.mu.Unlock()
	if up.hits["spare"] != 0 {
		t.Errorf("the spare was called %d times with its breaker open", up.hits["spare"])
	}
	if up.hits["main"] != 1 {
		t.Errorf("the chain was not used: %v", up.hits)
	}
}

// A failing spare must not surface to the client: the request lands on the next provider.
// The upstream call is real HTTP, so this covers the failover path end to end.
type twoUpstreams struct {
	mu     sync.Mutex
	hits   map[string]int
	broken string
	// status, when set, is returned for every request regardless of account: used to prove
	// how a whole class of upstream answer is classified per dialect.
	status int
	body   string
}

func (u *twoUpstreams) server(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		u.mu.Lock()
		name := strings.TrimPrefix(strings.Split(strings.TrimPrefix(r.URL.Path, "/"), "/")[0], "")
		u.hits[name]++
		broken := name == u.broken
		u.mu.Unlock()

		if u.status != 0 {
			w.WriteHeader(u.status)
			body := u.body
			if body == "" {
				body = `{"error":"upstream refused"}`
			}
			fmt.Fprint(w, body)
			return
		}
		if broken {
			w.WriteHeader(502)
			fmt.Fprint(w, `{"error":"quota exhausted"}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"choices":[{"message":{"content":"from %s"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":5,"completion_tokens":2}}`, name)
	}))
}

func TestFirstProviderFailsOverToTheChain(t *testing.T) {
	up := &twoUpstreams{hits: map[string]int{}, broken: "spare"}
	srv := up.server(t)
	t.Cleanup(srv.Close)

	prev := provs
	t.Cleanup(func() { provs = prev })
	provs = map[string]*Provider{
		"ollama-lenjan": {Base: srv.URL + "/spare", KeyEnv: "SPARE_KEY", First: true,
			Models: map[string]ModelRoute{"m": {ID: "spare-id"}}},
		"ollama": {Base: srv.URL + "/main", KeyEnv: "MAIN_KEY",
			Models: map[string]ModelRoute{"m": {ID: "main-id"}}},
	}
	t.Setenv("SPARE_KEY", "k1")
	t.Setenv("MAIN_KEY", "k2")
	led = NewLedger(t.TempDir() + "/spend.json")
	breakers = map[string]*breakerEntry{}
	shadow = false
	authToken = ""
	nowFn = func() time.Time { return mon(13) }

	router := httptest.NewServer(newMux())
	t.Cleanup(router.Close)

	resp, raw := postAnthropic(t, router.URL, "/v1/chat/completions",
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d: %s", resp.StatusCode, raw)
	}
	if !strings.Contains(string(raw), "from main") {
		t.Fatalf("answer did not come from the fallback: %s", raw)
	}
	if got := resp.Header.Get("X-Router-Failover"); got != "true" {
		t.Errorf("X-Router-Failover = %q, want true", got)
	}
	up.mu.Lock()
	defer up.mu.Unlock()
	if up.hits["spare"] != 1 || up.hits["main"] != 1 {
		t.Errorf("hits = %v, want the spare tried once then the chain", up.hits)
	}
}

// ── enrolment ───────────────────────────────────────────────────────────────

func enrolSetup(t *testing.T) (dir string) {
	t.Helper()
	dir = t.TempDir()
	oldToken, oldEnv, oldExit := enrolTokenPathVar, enrolEnvPathVar, exitProcess
	enrolTokenPathVar = filepath.Join(dir, "enrol.token")
	enrolEnvPathVar = filepath.Join(dir, ".env")
	// The handler ends the process so systemd reloads the environment; in a test that
	// would take the whole run down, so it is observed instead.
	exitProcess = func(int) {}
	t.Cleanup(func() {
		enrolTokenPathVar, enrolEnvPathVar, exitProcess = oldToken, oldEnv, oldExit
	})
	if err := os.WriteFile(enrolEnvPathVar, []byte("ROUTER_ADDR=1\nOLLAMA_API_KEY=existing\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestEnrolIsOffWithoutToken(t *testing.T) {
	enrolSetup(t)
	srv := httptest.NewServer(newMux())
	t.Cleanup(srv.Close)

	for _, path := range []string{"/enrol/", "/enrol/anything"} {
		resp, err := http.Get(srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != 404 {
			t.Errorf("%s: status = %d, want 404 when no token is live", path, resp.StatusCode)
		}
	}
}

func TestEnrolWritesKeyOnce(t *testing.T) {
	enrolSetup(t)
	token, err := mintEnrolToken()
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(newMux())
	t.Cleanup(srv.Close)

	// The form must not leak the key field's value, and must exist.
	form, err := http.Get(srv.URL + "/enrol/" + token)
	if err != nil {
		t.Fatal(err)
	}
	page := make([]byte, 4096)
	n, _ := form.Body.Read(page)
	form.Body.Close()
	if form.StatusCode != 200 || !strings.Contains(string(page[:n]), `name="key"`) {
		t.Fatalf("form status = %d", form.StatusCode)
	}

	resp, err := http.PostForm(srv.URL+"/enrol/"+token,
		url.Values{"key": {"lenjan-key-value"}})
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("post status = %d, want 200", resp.StatusCode)
	}

	env, err := os.ReadFile(enrolEnvPathVar)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(env), "OLLAMA_LENJAN_API_KEY=lenjan-key-value") {
		t.Errorf("key not written: %s", env)
	}
	// The other lines must survive: this file is the service's whole environment.
	if !strings.Contains(string(env), "OLLAMA_API_KEY=existing") ||
		!strings.Contains(string(env), "ROUTER_ADDR=1") {
		t.Errorf("existing lines were lost: %s", env)
	}
	info, _ := os.Stat(enrolEnvPathVar)
	if info.Mode().Perm() != 0o600 {
		t.Errorf("mode = %o, want 600", info.Mode().Perm())
	}

	// Second use of the same link must be refused.
	again, err := http.PostForm(srv.URL+"/enrol/"+token, url.Values{"key": {"other"}})
	if err != nil {
		t.Fatal(err)
	}
	again.Body.Close()
	if again.StatusCode != 404 {
		t.Errorf("replay status = %d, want 404", again.StatusCode)
	}
	env, _ = os.ReadFile(enrolEnvPathVar)
	if strings.Contains(string(env), "other") {
		t.Error("a replay overwrote the key")
	}
}

func TestEnrolRejectsBadInput(t *testing.T) {
	enrolSetup(t)
	token, err := mintEnrolToken()
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(newMux())
	t.Cleanup(srv.Close)

	cases := []struct{ name, key string }{
		{"kosong", ""},
		{"baris baru", "abc\ndef"},
		{"ruang", "abc def"},
		{"terlalu panjang", strings.Repeat("k", 600)},
	}
	for _, c := range cases {
		resp, err := http.PostForm(srv.URL+"/enrol/"+token, url.Values{"key": {c.key}})
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode == 200 {
			t.Errorf("%s: accepted, want rejection", c.name)
		}
	}
	// Nothing was written, and the link still works.
	env, _ := os.ReadFile(enrolEnvPathVar)
	if strings.Contains(string(env), "OLLAMA_LENJAN_API_KEY") {
		t.Errorf("a rejected paste still wrote the key: %s", env)
	}
	if _, err := os.Stat(enrolTokenPathVar); err != nil {
		t.Error("a rejected paste consumed the token")
	}
}

func TestEnrolWrongTokenIsNotFound(t *testing.T) {
	enrolSetup(t)
	if _, err := mintEnrolToken(); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(newMux())
	t.Cleanup(srv.Close)

	resp, err := http.PostForm(srv.URL+"/enrol/wrong-token", url.Values{"key": {"x"}})
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 404 {
		t.Errorf("status = %d, want 404", resp.StatusCode)
	}
	env, _ := os.ReadFile(enrolEnvPathVar)
	if strings.Contains(string(env), "OLLAMA_LENJAN") {
		t.Error("a wrong token wrote the key")
	}
}

// The key must never come back to the browser.
func TestEnrolNeverEchoesTheKey(t *testing.T) {
	enrolSetup(t)
	token, _ := mintEnrolToken()
	srv := httptest.NewServer(newMux())
	t.Cleanup(srv.Close)

	const secret = "SUPER-SECRET-LENJAN-KEY"
	resp, err := http.PostForm(srv.URL+"/enrol/"+token, url.Values{"key": {secret}})
	if err != nil {
		t.Fatal(err)
	}
	body := make([]byte, 8192)
	n, _ := resp.Body.Read(body)
	resp.Body.Close()
	if strings.Contains(string(body[:n]), secret) {
		t.Error("the response echoed the key back")
	}
	// And the key must not appear in any manifest the router serves.
	req, _ := http.NewRequest("GET", srv.URL+"/v1/models", nil)
	listing, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer listing.Body.Close()
	var payload any
	json.NewDecoder(listing.Body).Decode(&payload)
	if buf, _ := json.Marshal(payload); strings.Contains(string(buf), secret) {
		t.Error("the key appears in /v1/models")
	}
}
