package main

// Ollama chat dialect, request side: POST /api/chat and POST /api/generate.
// The router speaks OpenAI upstream, so an Ollama request is translated into an
// OpenAI chat payload here and the answer is translated back in ollama_render.go.

import (
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

type ollamaMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	Images    []string         `json:"images,omitempty"`
	ToolCalls []map[string]any `json:"tool_calls,omitempty"`
}

type ollamaChatRequest struct {
	Model    string           `json:"model"`
	Messages []ollamaMessage  `json:"messages"`
	Stream   *bool            `json:"stream"`
	Options  map[string]any   `json:"options"`
	Tools    []map[string]any `json:"tools"`
	System   string           `json:"system"`
	Prompt   string           `json:"prompt"`
	Format   any              `json:"format"`
}

// streams defaults to true, which is what Ollama does when the field is absent.
func (o *ollamaChatRequest) streams() bool {
	if o.Stream == nil {
		return true
	}
	return *o.Stream
}

// toOpenAIChat turns either dialect into an OpenAI chat payload.
func (o *ollamaChatRequest) toOpenAIChat(kind string) map[string]any {
	msgs := make([]map[string]any, 0, len(o.Messages)+2)
	if o.System != "" {
		msgs = append(msgs, map[string]any{"role": "system", "content": o.System})
	}
	for _, m := range o.Messages {
		role := m.Role
		if role == "" {
			role = "user"
		}
		if len(m.Images) > 0 && role == "user" {
			// Ollama carries images as bare base64; OpenAI wants a data URL, so the
			// media type has to be sniffed or the upstream call fails every time.
			parts := []any{map[string]any{"type": "text", "text": m.Content}}
			for _, img := range m.Images {
				parts = append(parts, map[string]any{
					"type":      "image_url",
					"image_url": map[string]any{"url": imageDataURL(img)},
				})
			}
			msgs = append(msgs, map[string]any{"role": role, "content": parts})
			continue
		}
		entry := map[string]any{"role": role, "content": m.Content}
		if len(m.ToolCalls) > 0 {
			entry["tool_calls"] = m.ToolCalls
		}
		msgs = append(msgs, entry)
	}
	if kind == "generate" {
		// /api/generate has no messages array; the prompt is a single user turn.
		msgs = append(msgs, map[string]any{"role": "user", "content": o.Prompt})
	}

	payload := map[string]any{"messages": msgs}
	if len(o.Tools) > 0 {
		payload["tools"] = o.Tools
	}
	switch f := o.Format.(type) {
	case string:
		if f == "json" {
			payload["response_format"] = map[string]any{"type": "json_object"}
		}
	case map[string]any:
		// A schema-bearing format is a JSON-mode request with structure; the
		// upstream only needs to know it must emit JSON.
		payload["response_format"] = map[string]any{"type": "json_object"}
	}
	applyOllamaOptions(payload, o.Options)
	return payload
}

// applyOllamaOptions maps the sampling knobs a client actually sets. Unknown
// options are dropped rather than forwarded: an OpenAI upstream rejects foreign
// keys with a 400 and the failure would be blamed on the provider.
func applyOllamaOptions(payload map[string]any, opts map[string]any) {
	if len(opts) == 0 {
		return
	}
	for from, to := range map[string]string{
		"temperature": "temperature",
		"top_p":       "top_p",
		"seed":        "seed",
		"num_predict": "max_tokens",
		"stop":        "stop",
	} {
		if v, ok := opts[from]; ok {
			payload[to] = v
		}
	}
}

func imageDataURL(b64 string) string {
	mime := "image/jpeg"
	if raw, err := base64.StdEncoding.DecodeString(b64); err == nil && len(raw) > 8 {
		switch {
		case raw[0] == 0x89 && raw[1] == 'P':
			mime = "image/png"
		case raw[0] == 'G' && raw[1] == 'I':
			mime = "image/gif"
		case raw[0] == 'R' && raw[1] == 'I':
			mime = "image/webp"
		}
	}
	return "data:" + mime + ";base64," + b64
}

func handleOllamaChat(w http.ResponseWriter, r *http.Request) {
	handleOllamaTalk(w, r, "chat")
}

func handleOllamaGenerate(w http.ResponseWriter, r *http.Request) {
	handleOllamaTalk(w, r, "generate")
}

// handleOllamaTalk resolves the model, then hands off to the shared failover
// loop so this dialect routes exactly like /v1/chat/completions.
func handleOllamaTalk(w http.ResponseWriter, r *http.Request, kind string) {
	var req ollamaChatRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 32<<20)).Decode(&req); err != nil {
		writeJSON(w, 400, map[string]any{"error": "bad json: " + err.Error()})
		return
	}
	model, ok := resolveModel(req.Model)
	if !ok {
		name := req.Model
		if name == "" {
			name = "(empty)"
		}
		writeJSON(w, 404, map[string]any{"error": "model " + name + " not found, try pulling it first"})
		return
	}
	if api := modelAPI(model); api != APIChat {
		writeJSON(w, 400, map[string]any{"error": model + " is a " + api +
			" model and is not available on the chat API"})
		return
	}
	serveOllama(w, r, req.toOpenAIChat(kind), req.streams(), model, req.Model, kind)
}

// handleOllamaShow answers /api/show from the model table: there is no local
// modelfile, but a client inspecting a model must not get a 404.
func handleOllamaShow(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model string `json:"model"`
		Name  string `json:"name"`
	}
	_ = json.NewDecoder(io.LimitReader(r.Body, 1<<16)).Decode(&req)
	want := req.Model
	if want == "" {
		want = req.Name
	}
	model, ok := resolveModel(want)
	if !ok {
		writeJSON(w, 404, map[string]any{"error": "model " + strings.TrimSpace(want) + " not found"})
		return
	}
	var by []string
	for name := range provs {
		if servesModel(name, model) {
			by = append(by, name)
		}
	}
	writeJSON(w, 200, map[string]any{
		"modelfile":  "# routed by offpeak-router; there is no local model file",
		"parameters": "",
		"template":   "",
		"details": map[string]any{
			"parent_model": "", "format": "api", "family": modelAPI(model),
			"families": by, "parameter_size": "", "quantization_level": "",
			"router_api": modelAPI(model), "router_providers": by,
		},
		"model_info": map[string]any{"router.owned_by": "offpeak-router"},
	})
}

// handleOllamaPs reports nothing resident: every request goes to an upstream.
func handleOllamaPs(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, 200, map[string]any{"models": []any{}})
}
