package main

// A thinking model puts its tokens in reasoning_content before any content.
// With a small budget the reasoning is the whole reply, so a translation that
// only reads content reports an empty answer for a request that worked.

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func setupWithUpstream(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	up := httptest.NewServer(handler)
	t.Cleanup(up.Close)

	t.Setenv("OPENCODE_API_KEY", "k-go")
	origProvs := provs
	t.Cleanup(func() { provs = origProvs })
	provs = map[string]*Provider{
		"go": {Base: up.URL + "/go/v1", KeyEnv: "OPENCODE_API_KEY",
			Models: map[string]ModelRoute{"mock-model": {ID: "mock-go"}}},
	}
	t.Cleanup(func() {
		led = NewLedger(t.TempDir() + "/spend.json")
		breakers = map[string]*breakerEntry{}
		authToken = ""
		shadow = false
	})

	led = NewLedger(t.TempDir() + "/spend.json")
	breakers = map[string]*breakerEntry{}
	authToken = ""
	shadow = false

	router := httptest.NewServer(newMux())
	t.Cleanup(router.Close)
	return router
}

func TestOllamaChatKeepsReasoningAsThinking(t *testing.T) {
	router := setupWithUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"content":"","reasoning_content":"weighing options"},`+
			`"finish_reason":"length"}],"usage":{"prompt_tokens":30,"completion_tokens":16}}`)
	})

	resp, body := postJSON(t, router.URL+"/api/chat",
		`{"model":"mock-model","stream":false,"messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, body = %s", resp.StatusCode, body)
	}

	var out struct {
		Message struct {
			Content  string `json:"content"`
			Thinking string `json:"thinking"`
		} `json:"message"`
		DoneReason string `json:"done_reason"`
		Eval       int    `json:"eval_count"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatal(err)
	}
	if out.Message.Thinking != "weighing options" {
		t.Errorf("reasoning was dropped: thinking = %q", out.Message.Thinking)
	}
	if out.Message.Content != "" {
		t.Errorf("content should stay empty here, got %q", out.Message.Content)
	}
	// The budget ran out while thinking, and a client needs to know why it got
	// no prose rather than assuming the server broke.
	if out.DoneReason != "length" {
		t.Errorf("done_reason = %q, want length", out.DoneReason)
	}
	if out.Eval != 16 {
		t.Errorf("eval_count = %d, want 16", out.Eval)
	}
}

func TestOllamaChatStreamEmitsThinkingChunks(t *testing.T) {
	router := setupWithUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hmm\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\" ok\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Blue\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":3}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	})

	resp, body := postJSON(t, router.URL+"/api/chat",
		`{"model":"mock-model","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, body = %s", resp.StatusCode, body)
	}

	var thinking, content strings.Builder
	lines := nonEmptyLines(string(body))
	for _, l := range lines {
		var chunk struct {
			Message struct {
				Content  string `json:"content"`
				Thinking string `json:"thinking"`
			} `json:"message"`
		}
		if err := json.Unmarshal([]byte(l), &chunk); err != nil {
			t.Fatalf("bad NDJSON line %q: %v", l, err)
		}
		thinking.WriteString(chunk.Message.Thinking)
		content.WriteString(chunk.Message.Content)
	}
	if thinking.String() != "hmm ok" {
		t.Errorf("thinking stream = %q, want %q", thinking.String(), "hmm ok")
	}
	if content.String() != "Blue" {
		t.Errorf("content stream = %q, want Blue", content.String())
	}
	// Without the reasoning chunks this reply would look like one lone token.
	if len(lines) < 3 {
		t.Errorf("only %d chunks reached the client, reasoning was dropped", len(lines))
	}
}

// A chunk carrying neither content nor reasoning must not be emitted: Ollama
// clients render every chunk and would show empty output.
func TestOllamaChatStreamSkipsEmptyChunks(t *testing.T) {
	router := setupWithUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	})

	_, body := postJSON(t, router.URL+"/api/chat",
		`{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"stream":true}`)
	var withContent int
	for _, l := range nonEmptyLines(string(body)) {
		var chunk struct {
			Done    bool `json:"done"`
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		}
		if json.Unmarshal([]byte(l), &chunk) != nil {
			continue
		}
		if !chunk.Done && chunk.Message.Content != "" {
			withContent++
		}
	}
	if withContent != 1 {
		t.Errorf("want exactly 1 content chunk, got %d in %q", withContent, body)
	}
}
