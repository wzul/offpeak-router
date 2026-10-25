package main

// Ollama-native shims: GET /api/tags and GET /api/version.
//
// A client pointed at an OpenAI-compatible server it cannot identify first probes
// the Ollama endpoints (Hermes accepts {server}/api/tags whose body carries a
// "models" key, and LM Studio answers it too). Without these routes that probe
// gets a 404 and the client gives up on the server, even though /v1/models would
// have answered. Both are read-only views over the same model table the chat
// routes use, so they can never advertise a model the router would reject.

import (
	"net/http"
	"sort"
	"strings"
	"time"
)

// startedAt keeps modified_at stable between calls instead of drifting.
var startedAt = time.Now().UTC().Format(time.RFC3339Nano)

type ollamaModel struct {
	Name       string         `json:"name"`
	Model      string         `json:"model"`
	ModifiedAt string         `json:"modified_at"`
	Size       int64          `json:"size"`
	Digest     string         `json:"digest"`
	Details    map[string]any `json:"details"`
}

func handleOllamaTags(w http.ResponseWriter, _ *http.Request) {
	ids := availableModels()
	sort.Strings(ids)

	models := make([]ollamaModel, 0, len(ids))
	for _, id := range ids {
		var by []string
		for name := range provs {
			if servesModel(name, id) {
				by = append(by, name)
			}
		}
		sort.Strings(by)

		// Ollama reports names as name:tag; publish the bare id plus :latest so a
		// client that appends a tag still round-trips to a model we serve.
		name := id
		if !strings.Contains(name, ":") {
			name += ":latest"
		}
		models = append(models, ollamaModel{
			Name:       name,
			Model:      name,
			ModifiedAt: startedAt,
			// A routed model has no local weights, so size and digest stay empty
			// rather than pretending to describe a blob that does not exist.
			Size:   0,
			Digest: "",
			Details: map[string]any{
				"parent_model":       "",
				"format":             "api",
				"family":             modelAPI(id),
				"families":           by,
				"parameter_size":     "",
				"quantization_level": "",
				"router_api":         modelAPI(id),
				"router_providers":   by,
			},
		})
	}
	writeJSON(w, 200, map[string]any{"models": models})
}

func handleOllamaVersion(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, 200, map[string]any{"version": Version})
}
