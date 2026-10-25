package main

import (
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

// A client speaking Ollama must get Ollama-shaped answers, or it parses nothing
// and reports an empty reply instead of an error.
func TestOllamaChatNonStream(t *testing.T) {
	up, srv := newUpstream(t)
	router := setup(t, srv)

	resp, body := postJSON(t, router.URL+"/api/chat",
		`{"model":"mock-model","stream":false,"messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, body = %s", resp.StatusCode, body)
	}
	if got := resp.Header.Get("X-Router-Provider"); got != "go" {
		t.Errorf("X-Router-Provider = %q, want go", got)
	}
	if got := resp.Header.Get("Content-Type"); !strings.Contains(got, "application/json") {
		t.Errorf("Content-Type = %q, streaming answer with stream:false", got)
	}

	var out struct {
		Model      string `json:"model"`
		CreatedAt  string `json:"created_at"`
		Done       bool   `json:"done"`
		DoneReason string `json:"done_reason"`
		Message    struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		PromptEval int `json:"prompt_eval_count"`
		Eval       int `json:"eval_count"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("decode: %v (%s)", err, body)
	}
	if !out.Done {
		t.Error("done must be true on a complete reply")
	}
	if out.Message.Role != "assistant" || out.Message.Content != "hi" {
		t.Errorf("message = %+v, want assistant/hi", out.Message)
	}
	if out.Model != "mock-model:latest" {
		t.Errorf("model = %q, want the client's name in name:tag form", out.Model)
	}
	if out.CreatedAt == "" {
		t.Error("created_at is required by the Ollama schema")
	}
	// Billing must still happen on this dialect, so the token counts have to
	// survive the translation.
	if out.PromptEval != 1000 || out.Eval != 100 {
		t.Errorf("token counts = %d/%d, want 1000/100", out.PromptEval, out.Eval)
	}
	if len(up.calls()) == 0 {
		t.Error("no upstream call was made")
	}
}

// Ollama streams unless told otherwise, so a missing stream field must not
// produce a single JSON document.
func TestOllamaChatStreamsByDefault(t *testing.T) {
	_, srv := newUpstream(t)
	router := setup(t, srv)

	resp, body := postJSON(t, router.URL+"/api/chat",
		`{"model":"mock-model","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, body = %s", resp.StatusCode, body)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "ndjson") {
		t.Errorf("Content-Type = %q, want application/x-ndjson", ct)
	}

	lines := nonEmptyLines(string(body))
	if len(lines) < 2 {
		t.Fatalf("want at least a chunk and a final line, got %d: %q", len(lines), body)
	}
	var first struct {
		Done    bool `json:"done"`
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	}
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatalf("first line is not JSON: %v (%q)", err, lines[0])
	}
	if first.Done {
		t.Error("a content chunk must not be marked done")
	}
	if first.Message.Content != "hi" {
		t.Errorf("first chunk content = %q, want hi", first.Message.Content)
	}

	var last struct {
		Done       bool   `json:"done"`
		DoneReason string `json:"done_reason"`
		Eval       int    `json:"eval_count"`
	}
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &last); err != nil {
		t.Fatalf("final line is not JSON: %v (%q)", err, lines[len(lines)-1])
	}
	if !last.Done {
		t.Error("the last line must carry done:true or a client waits forever")
	}
	if last.Eval != 100 {
		t.Errorf("final eval_count = %d, want 100 from the usage chunk", last.Eval)
	}
	// An SSE "data: " prefix would be unparseable for an Ollama client.
	if strings.Contains(string(body), "data: ") {
		t.Error("SSE framing leaked into the NDJSON stream")
	}
}

func TestOllamaGenerateUsesPromptField(t *testing.T) {
	up, srv := newUpstream(t)
	router := setup(t, srv)

	resp, body := postJSON(t, router.URL+"/api/generate",
		`{"model":"mock-model","prompt":"say hi","stream":false}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, body = %s", resp.StatusCode, body)
	}
	var out struct {
		Response string `json:"response"`
		Done     bool   `json:"done"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatal(err)
	}
	if out.Response != "hi" || !out.Done {
		t.Errorf("generate reply = %+v, want response=hi done=true", out)
	}
	if len(up.calls()) == 0 {
		t.Error("generate did not reach an upstream")
	}
}

// A decision model has no chat endpoint upstream; the Ollama route must refuse
// it exactly like /v1 does instead of forwarding and failing on every provider.
func TestOllamaChatRejectsDecisionModel(t *testing.T) {
	_, srv := newUpstream(t)
	router := setup(t, srv)

	orig := provs
	t.Cleanup(func() { provs = orig })
	provs = map[string]*Provider{
		"go": {Base: srv.URL + "/go/v1", KeyEnv: "OPENCODE_API_KEY",
			Models: map[string]ModelRoute{
				"mock-model": {ID: "mock-go"},
				"mock-clef":  {ID: "clef:27b", API: APISystemOne},
			}},
	}

	resp, body := postJSON(t, router.URL+"/api/chat",
		`{"model":"mock-clef","messages":[{"role":"user","content":"hi"}],"stream":false}`)
	if resp.StatusCode != 400 {
		t.Fatalf("status = %d, want 400; body = %s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "systemone") {
		t.Errorf("error should name the correct API, got %s", body)
	}
}

// Ollama answers 404 for an unpulled model, and clients show that message.
func TestOllamaChatUnknownModel(t *testing.T) {
	_, srv := newUpstream(t)
	router := setup(t, srv)

	resp, body := postJSON(t, router.URL+"/api/chat",
		`{"model":"nope:latest","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 404 {
		t.Fatalf("status = %d, want 404; body = %s", resp.StatusCode, body)
	}
}

func TestOllamaChatRequiresAuth(t *testing.T) {
	_, srv := newUpstream(t)
	router := setup(t, srv)
	authToken = "secret"
	t.Cleanup(func() { authToken = "" })

	resp, _ := postJSON(t, router.URL+"/api/chat",
		`{"model":"mock-model","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 401 {
		t.Errorf("status = %d, want 401", resp.StatusCode)
	}
}

func TestToOpenAIChatMapping(t *testing.T) {
	stream := false
	req := ollamaChatRequest{
		Model:  "m",
		System: "be brief",
		Messages: []ollamaMessage{
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "hello"},
		},
		Prompt:  "unused for chat",
		Stream:  &stream,
		Options: map[string]any{"temperature": 0.4, "num_predict": 64, "top_k": 40},
		Format:  "json",
	}
	payload := req.toOpenAIChat("chat")

	msgs, _ := payload["messages"].([]map[string]any)
	if len(msgs) != 3 {
		t.Fatalf("want system + 2 turns, got %d", len(msgs))
	}
	if msgs[0]["role"] != "system" || msgs[0]["content"] != "be brief" {
		t.Errorf("system prompt was not prepended: %+v", msgs[0])
	}
	if payload["temperature"] != 0.4 {
		t.Errorf("temperature not mapped: %v", payload["temperature"])
	}
	if payload["max_tokens"] != 64 {
		t.Errorf("num_predict must map to max_tokens, got %v", payload["max_tokens"])
	}
	// top_k has no OpenAI equivalent: forwarding it earns a 400 from upstream.
	if _, ok := payload["top_k"]; ok {
		t.Error("unknown option top_k was forwarded")
	}
	if rf, _ := payload["response_format"].(map[string]any); rf["type"] != "json_object" {
		t.Errorf(`format:"json" must set response_format json_object, got %v`, payload["response_format"])
	}

	// /api/generate has no messages: the prompt becomes the user turn.
	gen := (&ollamaChatRequest{Model: "m", Prompt: "say hi"}).toOpenAIChat("generate")
	gm, _ := gen["messages"].([]map[string]any)
	if len(gm) != 1 || gm[0]["content"] != "say hi" {
		t.Fatalf("generate prompt not mapped: %+v", gm)
	}
}

func TestOllamaStreamsDefaultsToTrue(t *testing.T) {
	var req ollamaChatRequest
	if !req.streams() {
		t.Error("absent stream field must default to true like Ollama")
	}
	no := false
	req.Stream = &no
	if req.streams() {
		t.Error("explicit stream:false was ignored")
	}
}

func TestImageDataURLSniffsType(t *testing.T) {
	png := base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\nrest"))
	if got := imageDataURL(png); !strings.HasPrefix(got, "data:image/png;base64,") {
		t.Errorf("png not detected: %q", got[:30])
	}
	jpg := base64.StdEncoding.EncodeToString([]byte("\xff\xd8\xff\xe0rest"))
	if got := imageDataURL(jpg); !strings.HasPrefix(got, "data:image/jpeg;base64,") {
		t.Errorf("jpeg default broken: %q", got[:30])
	}
}

func TestOllamaShowAndPs(t *testing.T) {
	_, srv := newUpstream(t)
	router := setup(t, srv)

	resp, body := postJSON(t, router.URL+"/api/show", `{"model":"mock-model"}`)
	if resp.StatusCode != 200 {
		t.Fatalf("/api/show status = %d, body = %s", resp.StatusCode, body)
	}
	var show struct {
		Details map[string]any `json:"details"`
	}
	if err := json.Unmarshal(body, &show); err != nil {
		t.Fatal(err)
	}
	if show.Details["family"] != APIChat {
		t.Errorf("show details.family = %v, want %q", show.Details["family"], APIChat)
	}

	resp, body = postJSON(t, router.URL+"/api/ps", `{}`)
	if resp.StatusCode != 200 || !strings.Contains(string(body), `"models":[]`) {
		t.Errorf("/api/ps = %d %s, want an empty models list", resp.StatusCode, body)
	}
}

func postJSON(t *testing.T, url, payload string) (*http.Response, []byte) {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(payload))
	if err != nil {
		t.Fatalf("post %s: %v", url, err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	b, _ := io.ReadAll(resp.Body)
	return resp, b
}

func nonEmptyLines(s string) []string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if strings.TrimSpace(l) != "" {
			out = append(out, l)
		}
	}
	return out
}
