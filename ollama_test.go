package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// A model-listing probe (Hermes, LM Studio clients) accepts the server only if
// the body has a "models" key, so that key is the contract, not a nicety.
func TestOllamaTagsShape(t *testing.T) {
	prev := provs
	provs = map[string]*Provider{
		"go": {Model: "go-only", Models: map[string]ModelRoute{"deepseek-v4.1-flash": {ID: "go-id", API: APIChat}}},
		"oll": {Model: "oll-only", Models: map[string]ModelRoute{
			"deepseek-v4.1-flash": {ID: "oll-id", API: APIChat},
			"clef":                {ID: "clef:27b", API: APISystemOne},
		}},
	}
	defer func() { provs = prev }()

	rec := httptest.NewRecorder()
	handleOllamaTags(rec, httptest.NewRequest(http.MethodGet, "/api/tags", nil))
	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}

	var body struct {
		Models []struct {
			Name    string         `json:"name"`
			Model   string         `json:"model"`
			Size    int64          `json:"size"`
			Details map[string]any `json:"details"`
		} `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Models) != 2 {
		t.Fatalf("want both served models, got %d: %+v", len(body.Models), body.Models)
	}

	byName := map[string]int{}
	for i, m := range body.Models {
		byName[m.Name] = i
		if m.Name != m.Model {
			t.Errorf("name and model must match a client may use either: %q vs %q", m.Name, m.Model)
		}
		if !strings.Contains(m.Name, ":") {
			t.Errorf("name %q is not Ollama-shaped (needs name:tag)", m.Name)
		}
		if m.Size != 0 {
			t.Errorf("routed models have no local weights, size must be 0, got %d", m.Size)
		}
	}

	chat, ok := byName["deepseek-v4.1-flash:latest"]
	if !ok {
		t.Fatalf("chat model missing: %+v", byName)
	}
	if api := body.Models[chat].Details["router_api"]; api != APIChat {
		t.Errorf("router_api = %v, want %q", api, APIChat)
	}

	dec, ok := byName["clef:latest"]
	if !ok {
		t.Fatalf("decision model missing: %+v", byName)
	}
	// The advertised family must be the one a request has to use, or a client
	// will POST a decision model at /v1/chat/completions and get a 400.
	if api := body.Models[dec].Details["router_api"]; api != APISystemOne {
		t.Errorf("clef family = %v, want %q", api, APISystemOne)
	}
}

// Every advertised name must correspond to a model the router actually serves.
func TestOllamaTagsMatchesModelsRoute(t *testing.T) {
	prev := provs
	provs = map[string]*Provider{
		"go": {Model: "x", Models: map[string]ModelRoute{"mock-model": {ID: "x-id", API: APIChat}}},
	}
	defer func() { provs = prev }()

	rec := httptest.NewRecorder()
	handleOllamaTags(rec, httptest.NewRequest(http.MethodGet, "/api/tags", nil))
	var tags struct {
		Models []struct{ Name string } `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &tags); err != nil {
		t.Fatal(err)
	}

	rec2 := httptest.NewRecorder()
	handleModels(rec2, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	var list struct {
		Data []struct{ ID string } `json:"data"`
	}
	if err := json.Unmarshal(rec2.Body.Bytes(), &list); err != nil {
		t.Fatal(err)
	}

	if len(tags.Models) != len(list.Data) {
		t.Fatalf("tags lists %d models, /v1/models lists %d", len(tags.Models), len(list.Data))
	}
	for i, m := range list.Data {
		if !strings.HasPrefix(tags.Models[i].Name, m.ID) {
			t.Errorf("tag %q does not describe /v1/models id %q", tags.Models[i].Name, m.ID)
		}
	}
}

func TestOllamaVersionReportsBuild(t *testing.T) {
	rec := httptest.NewRecorder()
	handleOllamaVersion(rec, httptest.NewRequest(http.MethodGet, "/api/version", nil))
	var body map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["version"] != Version {
		t.Errorf("version = %q, want %q", body["version"], Version)
	}
}

func TestOllamaRoutesRequireAuth(t *testing.T) {
	prev := authToken
	authToken = "secret-token"
	defer func() { authToken = prev }()

	for _, path := range []string{"/api/tags", "/api/version"} {
		rec := httptest.NewRecorder()
		newMux().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
		if rec.Code != 401 {
			t.Errorf("%s without a token = %d, want 401", path, rec.Code)
		}
	}
}
