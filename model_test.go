package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// setupModels builds a three-provider world where one model is served by
// everybody and one only by Ollama — the production shape that motivated the
// model table: a client asks for a name, and only some providers know it.
func setupModels(t *testing.T, up *httptest.Server) *httptest.Server {
	t.Helper()
	t.Setenv("OPENCODE_API_KEY", "k-go")
	t.Setenv("OLLAMA_API_KEY", "k-ollama")
	t.Setenv("NOUS_API_KEY", "k-nous")

	origProvs := provs
	origModel := canonicalModel
	t.Cleanup(func() { provs = origProvs; canonicalModel = origModel })

	provs = map[string]*Provider{
		"go": {Base: up.URL + "/go/v1", KeyEnv: "OPENCODE_API_KEY",
			Models: map[string]ModelRoute{"alpha": {ID: "go-alpha"}}},
		"ollama": {Base: up.URL + "/ollama/v1", KeyEnv: "OLLAMA_API_KEY",
			Models: map[string]ModelRoute{
				"alpha": {ID: "oll-alpha"},
				// Priced apart from the provider default on purpose.
				"beta": {ID: "oll-beta",
					Rates: &Rates{In: 1.0, Cache: 1.0, Out: 1.0, PeakMult: 1.0}},
				// A decision model: /v1/systemone only, billed on input tokens.
				"gamma": {ID: "oll-gamma", API: APISystemOne,
					Rates: &Rates{In: 2.0, Cache: 2.0, Out: 2.0, PeakMult: 1.0}},
			}},
		"nous": {Base: up.URL + "/nous/v1", KeyEnv: "NOUS_API_KEY",
			Models: map[string]ModelRoute{"alpha": {ID: "nous-alpha"}}},
	}
	canonicalModel = "alpha"
	led = NewLedger(filepath.Join(t.TempDir(), "spend.json"))
	breakers = map[string]*breakerEntry{}
	shadow = false
	authToken = ""
	nowFn = func() time.Time { return mon(2) }

	s := httptest.NewServer(newMux())
	t.Cleanup(s.Close)
	return s
}

func postModel(t *testing.T, router *httptest.Server, model string) (*http.Response, []byte) {
	t.Helper()
	payload := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hi"}]}`, model)
	resp, err := http.Post(router.URL+"/v1/chat/completions",
		"application/json", strings.NewReader(payload))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	b, _ := io.ReadAll(resp.Body)
	return resp, b
}

func TestModelAliasResolves(t *testing.T) {
	_, up := newUpstream(t)
	setupModels(t, up)

	for _, tc := range []struct {
		in   string
		want string
		ok   bool
	}{
		{"alpha", "alpha", true}, // canonical name
		{"beta", "beta", true},   // canonical name
		{"oll-beta", "beta", true},
		{"go-alpha", "alpha", true},
		{"nous-alpha", "alpha", true},
		{"", "alpha", true}, // default
		{"clef", "", false},
	} {
		got, ok := resolveModel(tc.in)
		if ok != tc.ok || got != tc.want {
			t.Errorf("resolveModel(%q) = (%q,%v), want (%q,%v)",
				tc.in, got, ok, tc.want, tc.ok)
		}
	}
}

// A model only one provider serves must not be offered to the others, and a
// shared model must still see every provider. Without the model filter in
// candidateOrder both cases collapse into "everybody".
func TestCandidateOrderFilteredByModel(t *testing.T) {
	_, up := newUpstream(t)
	setupModels(t, up)
	fresh := map[string]float64{"go": 0, "ollama": 0, "nous": 0}

	if got := candidateOrder(mon(2), fresh, "beta"); len(got) != 1 || got[0] != "ollama" {
		t.Errorf("beta order = %v, want exactly [ollama]", got)
	}
	if got := candidateOrder(mon(2), fresh, "alpha"); len(got) != 3 {
		t.Errorf("alpha order = %v, want all three providers", got)
	}
	if got := candidateOrder(mon(2), fresh, ""); len(got) != 3 {
		t.Errorf("unfiltered order = %v, want all three providers", got)
	}
}

// The request that used to be impossible: ask for a model only Ollama has and
// have the router actually send Ollama's id for it.
func TestRoutesUnsheredModelToItsOnlyProvider(t *testing.T) {
	u, up := newUpstream(t)
	router := setupModels(t, up)

	resp, body := postModel(t, router, "beta")
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, body = %s", resp.StatusCode, body)
	}
	if got := u.calls(); len(got) != 1 || got[0] != "ollama" {
		t.Fatalf("upstream calls = %v, want exactly [ollama]", got)
	}
	if !strings.Contains(string(body), `"echo_model":"oll-beta"`) {
		t.Errorf("upstream model id not translated: %s", body)
	}
	if got := resp.Header.Get("X-Router-Provider"); got != "ollama" {
		t.Errorf("X-Router-Provider = %q, want ollama", got)
	}
	if got := resp.Header.Get("X-Router-Model"); got != "beta" {
		t.Errorf("X-Router-Model = %q, want beta", got)
	}
}

// Asking for a provider's private id is an alias, not a distinct model: the
// router must still route on the canonical name.
func TestUpstreamIdAcceptedAsAlias(t *testing.T) {
	u, up := newUpstream(t)
	router := setupModels(t, up)

	resp, body := postModel(t, router, "oll-alpha")
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, body = %s", resp.StatusCode, body)
	}
	if got := resp.Header.Get("X-Router-Model"); got != "alpha" {
		t.Errorf("X-Router-Model = %q, want alpha (alias resolved)", got)
	}
	for _, c := range u.calls() {
		if c == "oll-alpha" {
			t.Errorf("upstream path leaked the alias: %v", u.calls())
		}
	}
}

// An unknown model must fail loudly and list what exists. Silently serving a
// different model was the old behaviour and is the bug this replaces.
func TestUnknownModelRejected(t *testing.T) {
	u, up := newUpstream(t)
	router := setupModels(t, up)

	resp, body := postModel(t, router, "clef")
	if resp.StatusCode != 400 {
		t.Fatalf("status = %d, want 400: %s", resp.StatusCode, body)
	}
	if got := u.calls(); len(got) != 0 {
		t.Errorf("upstreams were called for an unknown model: %v", got)
	}
	var env struct {
		Error struct {
			Available []string `json:"available"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatalf("bad error body: %v", err)
	}
	if len(env.Error.Available) != 3 {
		t.Errorf("available = %v, want alpha, beta and gamma", env.Error.Available)
	}
}

// When the only provider serving a model is down, the router must fail the
// request — not quietly answer with a different model from another provider.
func TestNoSilentSubstitutionWhenOnlyProviderFails(t *testing.T) {
	u, up := newUpstream(t)
	router := setupModels(t, up)
	u.fail["ollama"] = true

	resp, body := postModel(t, router, "beta")
	if resp.StatusCode != 502 {
		t.Fatalf("status = %d, want 502: %s", resp.StatusCode, body)
	}
	for _, c := range u.calls() {
		if c != "ollama" {
			t.Errorf("fell back to %q for a model it cannot serve: %v", c, u.calls())
		}
	}
}

// A per-model rate override must win over the provider default, otherwise a
// cheap and an expensive model on the same provider are accounted as one price.
func TestPerModelRatesOverrideProviderRates(t *testing.T) {
	_, up := newUpstream(t)
	setupModels(t, up)

	// 1M input tokens at the provider default for ollama ($0.15/M).
	if got := requestCost("ollama", "alpha", 1e6, 0, 0, false); got != 0.15 {
		t.Errorf("alpha cost = %v, want 0.15 (provider rates)", got)
	}
	// The same tokens on the overridden model ($1.00/M).
	if got := requestCost("ollama", "beta", 1e6, 0, 0, false); got != 1.0 {
		t.Errorf("beta cost = %v, want 1.0 (model override)", got)
	}
	// An unknown model falls back to provider rates rather than to zero.
	if got := requestCost("ollama", "ghost", 1e6, 0, 0, false); got != 0.15 {
		t.Errorf("unknown model cost = %v, want provider default 0.15", got)
	}
}

// The default model must remain servable when the client sends nothing, and the
// resolved name must be reported so a caller can see what it actually got.
func TestEmptyModelUsesDefault(t *testing.T) {
	_, up := newUpstream(t)
	router := setupModels(t, up)

	resp, body := postModel(t, router, "")
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, body = %s", resp.StatusCode, body)
	}
	if got := resp.Header.Get("X-Router-Model"); got != "alpha" {
		t.Errorf("X-Router-Model = %q, want alpha", got)
	}
}

// A decision model must not be offered on the chat route: upstream has no
// /chat/completions for it, so every attempt would be a wasted 400 that looks
// like a provider outage in the failover log.
func TestDecisionModelRejectedOnChatRoute(t *testing.T) {
	u, up := newUpstream(t)
	router := setupModels(t, up)

	resp, body := postModel(t, router, "gamma")
	if resp.StatusCode != 400 {
		t.Fatalf("status = %d, want 400: %s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "/v1/systemone") {
		t.Errorf("error does not point at the right endpoint: %s", body)
	}
	if got := u.calls(); len(got) != 0 {
		t.Errorf("upstreams were called for a chat request they cannot serve: %v", got)
	}
}

// The route that makes a decision model usable through the router at all.
func TestSystemOneRoutesAndBills(t *testing.T) {
	u, up := newUpstream(t)
	router := setupModels(t, up)

	payload := `{"model":"gamma","state":"Hello World",` +
		`"questions":{"hi":{"type":"noul","instructions":"Is it a greeting?"}}}`
	resp, err := http.Post(router.URL+"/v1/systemone", "application/json",
		strings.NewReader(payload))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, body = %s", resp.StatusCode, body)
	}
	if got := u.calls(); len(got) != 1 || got[0] != "ollama" {
		t.Fatalf("upstream calls = %v, want exactly [ollama]", got)
	}
	if got := resp.Header.Get("X-Router-Provider"); got != "ollama" {
		t.Errorf("X-Router-Provider = %q, want ollama", got)
	}
	if got := resp.Header.Get("X-Router-Model"); got != "gamma" {
		t.Errorf("X-Router-Model = %q, want gamma", got)
	}
	// 1000 input tokens at the model's own $2.00/M rate.
	if got := resp.Header.Get("X-Router-Usd"); got != "0.002000" {
		t.Errorf("X-Router-Usd = %q, want 0.002000", got)
	}
	if !strings.Contains(string(body), `"answers"`) {
		t.Errorf("upstream body not passed through: %s", body)
	}
}

// The mirror guard: a chat model on the decision route is a client mistake, not
// an upstream failure.
func TestChatModelRejectedOnSystemOneRoute(t *testing.T) {
	u, up := newUpstream(t)
	router := setupModels(t, up)

	payload := `{"model":"alpha","state":"x","questions":{}}`
	resp, err := http.Post(router.URL+"/v1/systemone", "application/json",
		strings.NewReader(payload))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 400 {
		t.Fatalf("status = %d, want 400: %s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "/v1/chat/completions") {
		t.Errorf("error does not point at the right endpoint: %s", body)
	}
	if got := u.calls(); len(got) != 0 {
		t.Errorf("upstreams were called for a decision request they cannot serve: %v", got)
	}
}

// The model list has to say which endpoint each model belongs to, otherwise a
// client cannot know where to send it.
func TestModelsListReportsAPI(t *testing.T) {
	_, up := newUpstream(t)
	router := setupModels(t, up)

	resp, err := http.Get(router.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var m struct{ Data []map[string]any }
	_ = json.NewDecoder(resp.Body).Decode(&m)

	api := map[string]string{}
	for _, d := range m.Data {
		id, _ := d["id"].(string)
		a, _ := d["api"].(string)
		api[id] = a
	}
	if api["gamma"] != APISystemOne {
		t.Errorf("gamma api = %q, want %q", api["gamma"], APISystemOne)
	}
	if api["alpha"] != APIChat || api["beta"] != APIChat {
		t.Errorf("chat models mislabelled: %v", api)
	}
}
