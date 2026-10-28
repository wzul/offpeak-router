package main

// An unpaid bill is not a hiccup. These tests pin down which failures are held, that a hold
// outlives any cooldown, and that the in-process probe is what brings a provider back - not
// a request, and not the passage of time on its own.

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestHoldsIndefinitelyClassification(t *testing.T) {
	held := []struct {
		name string
		err  error
	}{
		{"402", newUpstreamError(402, "payment required")},
		{"401 key mati", newUpstreamError(401, "invalid api key")},
		{"403 bil", newUpstreamError(403,
			`{"error":"your subscription payment is past due. update your payment method"}`)},
		{"403 kuota", newUpstreamError(403, "monthly quota exhausted")},
	}
	for _, c := range held {
		if hold, _ := holdsIndefinitely(c.err); !hold {
			t.Errorf("%s: want held indefinitely", c.name)
		}
	}

	transient := []struct {
		name string
		err  error
	}{
		{"429 kadar", newUpstreamError(429, "too many requests")},
		{"500", newUpstreamError(500, "internal")},
		{"503", newUpstreamError(503, "overloaded")},
		{"403 bukan bil", newUpstreamError(403, "blocked by policy")},
		{"bukan upstreamError", errFake("connection reset")},
	}
	for _, c := range transient {
		if hold, _ := holdsIndefinitely(c.err); hold {
			t.Errorf("%s: want the normal cooldown, not a hold", c.name)
		}
	}
}

type errFake string

func (e errFake) Error() string { return string(e) }

// A held provider must stay out however long the clock runs.
func TestHeldProviderStaysOutBeyondCooldown(t *testing.T) {
	oldBreakers, oldNow := breakers, nowFn
	t.Cleanup(func() { breakers, nowFn = oldBreakers, oldNow })
	breakers = map[string]*breakerEntry{}

	base := mon(13)
	now := base
	nowFn = func() time.Time { return now }

	breakerFailWith("ollama-lenjan", newUpstreamError(403, "payment past due"))
	if !breakerOpen("ollama-lenjan") {
		t.Fatal("a billing failure must take the provider out at once")
	}
	// Ten hours of virtual time: far beyond breakSeconds, and it must still be out.
	now = base.Add(10 * time.Hour)
	if !breakerOpen("ollama-lenjan") {
		t.Error("the hold expired on its own; a bill does not fix itself in 60 seconds")
	}
	// A transient failure, by contrast, must clear on its own. Reset the clock first: the
	// cooldown is measured from the failure, so leaving it ten hours ahead would make the
	// transient check meaningless.
	now = base
	breakerFailWith("go", newUpstreamError(429, "slow down"))
	for i := 0; i < breakAfter; i++ {
		breakerFailWith("go", newUpstreamError(429, "slow down"))
	}
	if !breakerOpen("go") {
		t.Fatal("repeated transient failures should open the breaker")
	}
	now = base.Add(2 * breakSeconds)
	if breakerOpen("go") {
		t.Error("a transient failure should clear after the cooldown")
	}
}

// The probe is what releases a hold, and it runs on its own schedule.
func TestProbeReleasesHoldOnlyOnSuccess(t *testing.T) {
	oldBreakers, oldNow, oldProbe := breakers, nowFn, probeProviderFn
	t.Cleanup(func() { breakers, nowFn, probeProviderFn = oldBreakers, oldNow, oldProbe })
	breakers = map[string]*breakerEntry{}

	now := mon(13)
	nowFn = func() time.Time { return now }
	breakerFailWith("ollama-lenjan", newUpstreamError(402, "payment required"))

	probes := 0
	healthy := false
	probeProviderFn = func(context.Context, string, *Provider) bool {
		probes++
		return healthy
	}

	// Not due yet: nothing is probed before the first gap has passed.
	runProbeRound(context.Background())
	if probes != 0 {
		t.Fatalf("probed %d times before the first gap elapsed", probes)
	}

	now = mon(13).Add(probeFirstGap)
	runProbeRound(context.Background()) // provider still broken
	if probes != 1 {
		t.Fatalf("probes = %d, want 1", probes)
	}
	if !breakerOpen("ollama-lenjan") {
		t.Fatal("a failing probe must not release the hold")
	}

	// The next probe is further away: a long dead account must not cost a call a minute.
	due := breakerProbeDue(now.Add(probeFirstGap))
	if len(due) != 0 {
		t.Errorf("still probing at the old interval: %v", due)
	}
	now = mon(13).Add(probeFirstGap + 2*probeFirstGap)
	healthy = true
	runProbeRound(context.Background())
	if breakerOpen("ollama-lenjan") {
		t.Fatal("a successful probe must release the hold")
	}
	if due := breakerProbeDue(now); len(due) != 0 {
		t.Errorf("a released provider is still being probed: %v", due)
	}
}

// The probe backoff must escalate and stop escalating.
func TestProbeBackoffEscalatesAndCaps(t *testing.T) {
	oldBreakers, oldNow, oldProbe := breakers, nowFn, probeProviderFn
	t.Cleanup(func() { breakers, nowFn, probeProviderFn = oldBreakers, oldNow, oldProbe })
	breakers = map[string]*breakerEntry{}

	now := mon(13)
	nowFn = func() time.Time { return now }
	probeProviderFn = func(context.Context, string, *Provider) bool { return false }
	breakerFailWith("ollama-lenjan", newUpstreamError(402, "payment required"))

	var gaps []time.Duration
	for i := 0; i < 12; i++ {
		now = now.Add(probeMaxGap) // always past the next probe
		runProbeRound(context.Background())
		brMu.Lock()
		gaps = append(gaps, breakers["ollama-lenjan"].probeGap)
		brMu.Unlock()
	}
	if gaps[0] <= probeFirstGap {
		t.Errorf("first retry gap = %s, want longer than %s", gaps[0], probeFirstGap)
	}
	last := gaps[len(gaps)-1]
	if last != probeMaxGap {
		t.Errorf("final gap = %s, want the %s cap", last, probeMaxGap)
	}
}

// A held provider must never be called by real traffic, and the chain must answer.
func TestHeldProviderIsSkippedByTraffic(t *testing.T) {
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

	breakerFailWith("ollama-lenjan", newUpstreamError(403, "subscription payment is past due"))

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
		t.Errorf("the held provider was called %d times", up.hits["spare"])
	}
	if up.hits["main"] != 1 {
		t.Errorf("the chain was not used: %v", up.hits)
	}
}

// The probe itself must read a real status code correctly: 200 releases, 402 does not.
func TestProbeProviderHTTPReadsStatus(t *testing.T) {
	status := 200
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer probe-key" {
			t.Errorf("Authorization = %q", got)
		}
		w.WriteHeader(status)
		w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	t.Cleanup(srv.Close)

	t.Setenv("PROBE_KEY", "probe-key")
	p := &Provider{Base: srv.URL, KeyEnv: "PROBE_KEY",
		Models: map[string]ModelRoute{canonicalModel: {ID: "some-id"}}}

	if !probeProviderHTTP(context.Background(), "p", p) {
		t.Error("a 200 probe should report available")
	}
	status = 402
	if probeProviderHTTP(context.Background(), "p", p) {
		t.Error("a 402 probe must report unavailable")
	}
	// A provider with no chat model cannot be probed, and must not be released by guesswork.
	noChat := &Provider{Base: srv.URL, KeyEnv: "PROBE_KEY",
		Models: map[string]ModelRoute{"clef": {ID: "clef:27b", API: APISystemOne}}}
	if probeProviderHTTP(context.Background(), "p", noChat) {
		t.Error("a provider with no chat model was reported available")
	}
}

// /status is where an operator looks; a hold must be visible and clearly not a cooldown.
func TestStatusShowsTheHold(t *testing.T) {
	oldBreakers, oldNow := breakers, nowFn
	t.Cleanup(func() { breakers, nowFn = oldBreakers, oldNow })
	breakers = map[string]*breakerEntry{}
	now := mon(13)
	nowFn = func() time.Time { return now }

	breakerFailWith("ollama-lenjan", newUpstreamError(402, "payment required"))
	snap := breakerSnapshot()
	entry, ok := snap["ollama-lenjan"].(map[string]any)
	if !ok {
		t.Fatalf("held provider missing from the snapshot: %v", snap)
	}
	if entry["held"] != true {
		t.Errorf("held flag missing: %v", entry)
	}
	if _, ok := entry["next_probe_s"]; !ok {
		t.Errorf("next probe is not visible: %v", entry)
	}
}

// Every dialect must classify a dead account the same way. A bill does not become a hiccup
// because the client chose the Anthropic or the Ollama route, and a per-dialect difference here
// means one client retries a dead account every minute while the others have moved on.
func TestBillingFailureHoldsOnEveryDialect(t *testing.T) {
	oldBreakers, oldNow, oldProvs := breakers, nowFn, provs
	t.Cleanup(func() { breakers, nowFn, provs = oldBreakers, oldNow, oldProvs })

	up := &twoUpstreams{hits: map[string]int{}, status: 402,
		body: `{"error":"your subscription payment is past due"}`}
	srv := up.server(t)
	t.Cleanup(srv.Close)

	provs = map[string]*Provider{
		"ollama-lenjan": {Base: srv.URL + "/spare", KeyEnv: "SPARE_KEY", First: true, Web: true,
			Models: map[string]ModelRoute{"m": {ID: "spare-id"}}},
		"ollama": {Base: srv.URL + "/main", KeyEnv: "MAIN_KEY", Web: true,
			Models: map[string]ModelRoute{"m": {ID: "main-id"}}},
	}
	t.Setenv("SPARE_KEY", "k1")
	t.Setenv("MAIN_KEY", "k2")
	led = NewLedger(t.TempDir() + "/spend.json")
	shadow = false
	authToken = ""
	nowFn = func() time.Time { return mon(13) }

	router := httptest.NewServer(newMux())
	t.Cleanup(router.Close)

	dialects := []struct{ name, path, body string }{
		{"openai", "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`},
		{"anthropic", "/v1/messages", `{"model":"m","max_tokens":32,"messages":[{"role":"user","content":"hi"}]}`},
		{"ollama", "/api/chat", `{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false}`},
	}
	for _, d := range dialects {
		breakers = map[string]*breakerEntry{}
		resp, raw := postAnthropic(t, router.URL, d.path, d.body)
		if resp.StatusCode < 400 {
			t.Errorf("%s: status = %d, want a failure; body = %s", d.name, resp.StatusCode, raw)
		}
		brMu.Lock()
		entry := breakers["ollama-lenjan"]
		brMu.Unlock()
		if entry == nil || !entry.held {
			t.Errorf("%s: a billing failure did not hold the account (entry = %+v)", d.name, entry)
		}
		// And it must be out well past the transient window.
		nowFn = func() time.Time { return mon(13).Add(2 * time.Hour) }
		if !breakerOpen("ollama-lenjan") {
			t.Errorf("%s: the hold expired after two hours", d.name)
		}
		nowFn = func() time.Time { return mon(13) }
	}
}
