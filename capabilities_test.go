package main

// Discovery must answer "can this route see images?" directly. Before these
// fields existed, /v1/models, /api/tags and /api/show all looked identical for a
// vision model and a text-only one, so an agent searching for image support
// found none and reported the router had no vision model - while the model
// behind it answered "Red" for a red PNG on two upstreams.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"
)

func TestModelsRoutePublishesCapabilities(t *testing.T) {
	prev := provs
	provs = map[string]*Provider{
		"go": {Model: "x", Models: map[string]ModelRoute{"deepseek-v4.1-flash": {ID: "x-id", API: APIChat}}},
	}
	t.Cleanup(func() { provs = prev })

	rec := httptest.NewRecorder()
	handleModels(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	var body struct {
		Data []struct {
			ID                  string              `json:"id"`
			Capabilities        []string            `json:"capabilities"`
			CapabilityProviders map[string][]string `json:"capability_providers"`
			InputModalities     []string            `json:"input_modalities"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Data) != 1 {
		t.Fatalf("want 1 model, got %d", len(body.Data))
	}
	m := body.Data[0]

	if !contains(m.Capabilities, "vision") {
		t.Errorf("capabilities = %v, a vision model must say so", m.Capabilities)
	}
	if !contains(m.InputModalities, "image") {
		t.Errorf("input_modalities = %v, want image listed", m.InputModalities)
	}
	// Which upstream proved it matters: an unverified provider is not the same
	// as a verified one, and a client routing there should know.
	if got := m.CapabilityProviders["vision"]; len(got) == 0 {
		t.Errorf("capability_providers.vision is empty: %v", m.CapabilityProviders)
	} else if contains(got, "nous") {
		t.Errorf("nous was never verified for vision, got %v", got)
	}
}

func TestShowRoutePublishesCapabilities(t *testing.T) {
	prev := provs
	provs = map[string]*Provider{
		"ollama": {Model: "x", Models: map[string]ModelRoute{"deepseek-v4.1-flash": {ID: "deepseek-v4.1-flash:1017", API: APIChat}}},
	}
	t.Cleanup(func() { provs = prev })

	rec := httptest.NewRecorder()
	handleOllamaShow(rec, httptest.NewRequest(http.MethodPost, "/api/show",
		strings.NewReader(`{"model":"deepseek-v4.1-flash"}`)))

	var body struct {
		Capabilities []string `json:"capabilities"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	// "capabilities" is Ollama's own field name; a client checks it before it
	// sends an image.
	if !contains(body.Capabilities, "vision") {
		t.Errorf("/api/show capabilities = %v, want vision", body.Capabilities)
	}
}

func TestTagsPublishCapabilities(t *testing.T) {
	prev := provs
	provs = map[string]*Provider{
		"go": {Model: "x", Models: map[string]ModelRoute{"deepseek-v4.1-flash": {ID: "x-id", API: APIChat}}},
	}
	t.Cleanup(func() { provs = prev })

	rec := httptest.NewRecorder()
	handleOllamaTags(rec, httptest.NewRequest(http.MethodGet, "/api/tags", nil))

	var body struct {
		Models []struct {
			Details map[string]any `json:"details"`
		} `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Models) != 1 {
		t.Fatalf("want 1 model, got %d", len(body.Models))
	}
	caps, _ := body.Models[0].Details["capabilities"].([]any)
	if len(caps) == 0 || !containsAny(caps, "vision") {
		t.Errorf("details.capabilities = %v, want vision", body.Models[0].Details["capabilities"])
	}
}

// A model with no verified capability entry must not inherit vision by accident.
func TestUnknownModelClaimsNoVision(t *testing.T) {
	if hasVision("some-text-only-model") {
		t.Error("a model with no capability entry must not be advertised as vision-capable")
	}
	if hasVision("clef") {
		t.Error("clef is a decision model and must not claim vision")
	}
	if hasVision("deepseek-v4.1-flash") != true {
		t.Error("deepseek-v4.1-flash is verified vision-capable")
	}
}

func contains(hay []string, needle string) bool {
	for _, h := range hay {
		if h == needle {
			return true
		}
	}
	return false
}

func containsAny(hay []any, needle string) bool {
	for _, h := range hay {
		if s, ok := h.(string); ok && s == needle {
			return true
		}
	}
	return false
}

// gemma4 exists on ollama only: go answers "Model is unavailable". Both the
// canonical name and ollama's own id must resolve, or a client carrying the
// upstream id gets a 400 for a model the router does serve.
func TestGemma4RoutingAndCapabilities(t *testing.T) {
	prev := provs
	t.Cleanup(func() { provs = prev })
	provs = defaultProviders()

	for _, ask := range []string{"gemma4", "gemma4:31b"} {
		got, ok := resolveModel(ask)
		if !ok || got != "gemma4" {
			t.Errorf("resolveModel(%q) = %q, %v; want gemma4", ask, got, ok)
		}
	}
	// Two ollama accounts serve it now (the spare is pinned first), and go must not:
	// a map range would make this assertion flaky.
	var by []string
	for name := range provs {
		if servesModel(name, "gemma4") {
			by = append(by, name)
		}
	}
	sort.Strings(by)
	if len(by) != 2 || by[0] != "ollama" || by[1] != "ollama-lenjan" {
		t.Fatalf("gemma4 providers = %v, want [ollama ollama-lenjan]", by)
	}
	if !hasVision("gemma4") {
		t.Error("gemma4 was verified vision-capable on ollama")
	}
	if providers := capabilityProvidersFor("gemma4")["vision"]; len(providers) != 1 || providers[0] != "ollama" {
		t.Errorf("vision providers = %v, want [ollama] only", providers)
	}
	if api := modelAPI("gemma4"); api != APIChat {
		t.Errorf("api = %q, want chat", api)
	}
}

// Tool calling is advertised because it was measured: every upstream below was asked
// directly with a tool definition and answered with tool_calls. A client that reads this
// flag and sends tools must not discover the gap at request time.
func TestToolsCapabilityIsPublished(t *testing.T) {
	for _, model := range []string{"deepseek-v4.1-flash", "gemma4"} {
		found := false
		for _, c := range capabilitiesFor(model) {
			if c == "tools" {
				found = true
			}
		}
		if !found {
			t.Errorf("%s does not advertise tools: %v", model, capabilitiesFor(model))
		}
		providers := capabilityProvidersFor(model)["tools"]
		if len(providers) == 0 {
			t.Errorf("%s advertises tools with no verified upstream", model)
		}
	}
	// Clef is a decision schema, not a chat completion: claiming tools there would invite
	// a request the upstream cannot interpret.
	for _, c := range capabilitiesFor("clef") {
		if c == "tools" {
			t.Error("clef advertises tools")
		}
	}
}

// Every provider named against a capability must actually serve that model. A stale name
// here is worse than a missing one: it tells a client to route somewhere that cannot answer.
func TestVerifiedProvidersServeTheModel(t *testing.T) {
	for model, caps := range capabilityProviders {
		for capability, providers := range caps {
			for _, name := range providers {
				p, ok := provs[name]
				if !ok {
					t.Errorf("%s/%s names unknown provider %q", model, capability, name)
					continue
				}
				if _, served := p.Models[model]; !served {
					t.Errorf("%s/%s names %q, which does not serve %s", model, capability, name, model)
				}
			}
		}
	}
}

// The claim must reach clients, not just live in a map.
func TestModelsRouteExposesToolCapability(t *testing.T) {
	router := httptest.NewServer(newMux())
	t.Cleanup(router.Close)
	resp, raw := postAnthropic(t, router.URL, "/v1/models", "")
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d: %s", resp.StatusCode, raw)
	}
	var got struct {
		Data []struct {
			ID                  string              `json:"id"`
			Capabilities        []string            `json:"capabilities"`
			CapabilityProviders map[string][]string `json:"capability_providers"`
		} `json:"data"`
	}
	if err := json.Unmarshal([]byte(raw), &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, m := range got.Data {
		if m.ID != "deepseek-v4.1-flash" {
			continue
		}
		hasTools := false
		for _, c := range m.Capabilities {
			if c == "tools" {
				hasTools = true
			}
		}
		if !hasTools {
			t.Fatalf("capabilities = %v, want tools", m.Capabilities)
		}
		if len(m.CapabilityProviders["tools"]) == 0 {
			t.Errorf("capability_providers has no tools entry: %v", m.CapabilityProviders)
		}
		return
	}
	t.Fatal("deepseek-v4.1-flash missing from /v1/models")
}
