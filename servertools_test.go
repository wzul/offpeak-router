package main

// Server tools are only usable if the router actually executes them. These tests run a
// model that asks to search, a fake search backend that answers, and then check both what
// the client received and what the model was told - a response that looks right while the
// model never saw a result is the failure this file exists to catch.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeHits is what the search answers in these tests. runSearch/parseDDG have their own
// tests; what is under test here is that the loop runs the search, filters it and shows
// both sides the same thing.
var fakeHits = []webHit{
	{Title: "Go Release Notes", URL: "https://go.dev/doc/devel/release.html",
		Description: "Go 1.27 is planned for release in February 2027.", Position: 1},
	{Title: "Blocked Source", URL: "https://example.org/blocked",
		Description: "Should be filtered out.", Position: 2},
}

// scriptedUpstream plays a model that asks for a search and then answers.
type scriptedUpstream struct {
	mu       sync.Mutex
	requests []map[string]any
	replies  []string // successive upstream bodies
	stream   []string // successive SSE bodies when the client streams
	streamed bool
	idx      int
}

func (s *scriptedUpstream) serve(t *testing.T, ddg string) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/go/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		s.mu.Lock()
		s.requests = append(s.requests, body)
		i := s.idx
		s.idx++
		streaming, _ := body["stream"].(bool)
		s.mu.Unlock()

		if streaming && len(s.stream) > 0 {
			w.Header().Set("Content-Type", "text/event-stream")
			payload := s.stream[min(i, len(s.stream)-1)]
			fmt.Fprint(w, payload)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if len(s.replies) == 0 {
			fmt.Fprint(w, `{"choices":[{"message":{"content":"no script"},"finish_reason":"stop"}]}`)
			return
		}
		fmt.Fprint(w, s.replies[min(i, len(s.replies)-1)])
	})
	return httptest.NewServer(mux)
}

func searchToolRequest(t *testing.T, routerURL string, extra map[string]any) map[string]any {
	t.Helper()
	body := map[string]any{
		"model":      "claude-sonnet-4-5-20250929",
		"max_tokens": 256,
		"tools": []any{
			map[string]any{"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
		},
		"messages": []any{map[string]any{"role": "user", "content": "When is Go 1.27 out?"}},
	}
	for k, v := range extra {
		body[k] = v
	}
	return body
}

// serverToolSetup wires a router whose search backend is the fake upstream host.
func serverToolSetup(t *testing.T, up *scriptedUpstream, upstreamBase string) *httptest.Server {
	t.Helper()
	orig := provs
	t.Cleanup(func() { provs = orig })
	provs = map[string]*Provider{
		"go": {Base: upstreamBase, KeyEnv: "OPENCODE_API_KEY", SessionHeader: true,
			Models: map[string]ModelRoute{
				canonicalModel: {ID: "mock-go"},
				"second-model": {ID: "mock-second"},
			}},
	}
	t.Setenv("OPENCODE_API_KEY", "k")
	t.Setenv("BRAVE_SEARCH_API_KEY", "")
	led = NewLedger(t.TempDir() + "/spend.json")
	breakers = map[string]*breakerEntry{}
	shadow = false
	authToken = ""
	nowFn = func() time.Time { return mon(13) }
	// Answer the search directly instead of going out to the network: the SSRF guard
	// refuses loopback on purpose and is not going to be relaxed for a test.
	oldSearch, oldFetch := searchForModel, fetchForModel
	searchForModel = func(_ context.Context, q string, limit int) ([]webHit, string, error) {
		if limit > 0 && len(fakeHits) > limit {
			return fakeHits[:limit], "fake", nil
		}
		return fakeHits, "fake", nil
	}
	fetchForModel = func(_ context.Context, target string) fetchDoc {
		return fetchDoc{URL: target, Title: "Fetched page", Content: "Fetched body text.",
			Status: 200}
	}
	t.Cleanup(func() { searchForModel, fetchForModel = oldSearch, oldFetch })
	srv := httptest.NewServer(newMux())
	t.Cleanup(srv.Close)
	return srv
}

const wantsSearchReply = `{"choices":[{"message":{"role":"assistant","content":"I'll look that up.",
  "tool_calls":[{"id":"call_1","type":"function","function":{"name":"web_search","arguments":"{\"query\":\"go 1.27 release\"}"}}]},
  "finish_reason":"tool_calls"}],"usage":{"prompt_tokens":40,"completion_tokens":12}}`

const answeredReply = `{"choices":[{"message":{"role":"assistant","content":"Go 1.27 is planned for February 2027."},
  "finish_reason":"stop"}],"usage":{"prompt_tokens":300,"completion_tokens":20}}`

// The heart of it: the model asks, the router searches, the model answers, and the client
// sees the search it never had to run.
func TestServerToolSearchRunsInsideTheRequest(t *testing.T) {
	up := &scriptedUpstream{replies: []string{wantsSearchReply, answeredReply}}
	srv := up.serve(t, "")
	t.Cleanup(srv.Close)
	router := serverToolSetup(t, up, srv.URL+"/go/v1")

	resp, raw := postAnthropic(t, router.URL, "/v1/messages",
		mustJSON(searchToolRequest(t, router.URL, nil)))
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d: %s", resp.StatusCode, raw)
	}
	var out struct {
		Content []map[string]any `json:"content"`
		Usage   struct {
			Input  int `json:"input_tokens"`
			Output int `json:"output_tokens"`
			Server struct {
				Searches int `json:"web_search_requests"`
			} `json:"server_tool_use"`
		} `json:"usage"`
		StopReason string `json:"stop_reason"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("decode: %v (%s)", err, raw)
	}

	var kinds []string
	for _, b := range out.Content {
		kinds = append(kinds, b["type"].(string))
	}
	want := []string{"text", "server_tool_use", "web_search_tool_result", "text"}
	if strings.Join(kinds, ",") != strings.Join(want, ",") {
		t.Fatalf("blocks = %v, want %v\n%s", kinds, want, raw)
	}

	// The query the model asked for must be the query that was searched.
	use := out.Content[1]
	if use["name"] != "web_search" {
		t.Errorf("server_tool_use name = %v", use["name"])
	}
	if q, _ := use["input"].(map[string]any)["query"].(string); !strings.Contains(q, "1.27") {
		t.Errorf("search input = %v, want the model's query", use["input"])
	}

	// The result block must carry real hits, with snippets the model can use later.
	results, _ := out.Content[2]["content"].([]any)
	if len(results) == 0 {
		t.Fatalf("no search results: %s", raw)
	}
	first, _ := results[0].(map[string]any)
	if !strings.Contains(fmt.Sprint(first["url"]), "go.dev") {
		t.Errorf("first result url = %v", first["url"])
	}
	if first["encrypted_content"] == "" {
		t.Errorf("result has no encrypted_content: a later turn loses the snippet")
	}

	if out.Content[3]["text"] == "" {
		t.Errorf("the answer is missing: %s", raw)
	}
	if out.StopReason != "end_turn" {
		t.Errorf("stop_reason = %q, want end_turn (server tools do not stop the turn)", out.StopReason)
	}
	if out.Usage.Server.Searches != 1 {
		t.Errorf("usage.server_tool_use.web_search_requests = %d, want 1", out.Usage.Server.Searches)
	}
	// Billing must cover both upstream calls, not just the last one.
	if out.Usage.Input < 340 {
		t.Errorf("input_tokens = %d, want both rounds summed (40 + 300)", out.Usage.Input)
	}
}

// Domain filters are part of the tool definition and must be applied to the results.
func TestServerToolHonoursBlockedDomains(t *testing.T) {
	up := &scriptedUpstream{replies: []string{wantsSearchReply, answeredReply}}
	srv := up.serve(t, "")
	t.Cleanup(srv.Close)
	router := serverToolSetup(t, up, srv.URL+"/go/v1")

	req := searchToolRequest(t, router.URL, map[string]any{
		"tools": []any{map[string]any{
			"type": "web_search_20250305", "name": "web_search",
			"blocked_domains": []string{"example.org"},
		}},
	})
	_, raw := postAnthropic(t, router.URL, "/v1/messages", mustJSON(req))

	if strings.Contains(string(raw), "example.org") {
		t.Errorf("blocked domain leaked into the results: %s", raw)
	}
	if !strings.Contains(string(raw), "go.dev") {
		t.Errorf("allowed result missing: %s", raw)
	}
}

// max_uses is a hard cap the client asked for; exceeding it is an error result, not a
// silent extra search.
func TestServerToolMaxUses(t *testing.T) {
	alwaysSearch := `{"choices":[{"message":{"role":"assistant","content":"",
	  "tool_calls":[{"id":"call_1","type":"function","function":{"name":"web_search","arguments":"{\"query\":\"again\"}"}}]},
	  "finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`
	up := &scriptedUpstream{replies: []string{alwaysSearch, alwaysSearch, answeredReply}}
	srv := up.serve(t, "")
	t.Cleanup(srv.Close)
	router := serverToolSetup(t, up, srv.URL+"/go/v1")

	req := searchToolRequest(t, router.URL, map[string]any{
		"tools": []any{map[string]any{
			"type": "web_search_20250305", "name": "web_search", "max_uses": 1}},
	})
	_, raw := postAnthropic(t, router.URL, "/v1/messages", mustJSON(req))

	if !strings.Contains(string(raw), "max_uses_exceeded") {
		t.Fatalf("second search should be refused with max_uses_exceeded: %s", raw)
	}
	var out struct {
		Usage struct {
			Server struct {
				Searches int `json:"web_search_requests"`
			} `json:"server_tool_use"`
		} `json:"usage"`
	}
	json.Unmarshal(raw, &out)
	if out.Usage.Server.Searches != 1 {
		t.Errorf("searches = %d, want exactly the allowed 1", out.Usage.Server.Searches)
	}
}

// A client function tool must still be handed back to the client, even when the same
// request also declares a server tool.
func TestClientToolStillReturnedAlongsideServerTool(t *testing.T) {
	mixed := `{"choices":[{"message":{"role":"assistant","content":"",
	  "tool_calls":[{"id":"call_a","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"/tmp/x\"}"}}]},
	  "finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`
	up := &scriptedUpstream{replies: []string{mixed}}
	srv := up.serve(t, "")
	t.Cleanup(srv.Close)
	router := serverToolSetup(t, up, srv.URL+"/go/v1")

	req := searchToolRequest(t, router.URL, map[string]any{
		"tools": []any{
			map[string]any{"type": "web_search_20250305", "name": "web_search"},
			map[string]any{"name": "read_file", "description": "read a file",
				"input_schema": map[string]any{"type": "object",
					"properties": map[string]any{"path": map[string]any{"type": "string"}}}},
		},
	})
	resp, raw := postAnthropic(t, router.URL, "/v1/messages", mustJSON(req))
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d: %s", resp.StatusCode, raw)
	}
	if !strings.Contains(string(raw), `"tool_use"`) || !strings.Contains(string(raw), "read_file") {
		t.Fatalf("client tool call lost: %s", raw)
	}
	if !strings.Contains(string(raw), `"stop_reason":"tool_use"`) {
		t.Errorf("stop_reason should be tool_use when the client must act: %s", raw)
	}
}

// The upstream must receive the search results as a tool message, otherwise the model
// answers from memory and the whole loop was wasted.
func TestModelSeesSearchResults(t *testing.T) {
	up := &scriptedUpstream{replies: []string{wantsSearchReply, answeredReply}}
	srv := up.serve(t, "")
	t.Cleanup(srv.Close)
	router := serverToolSetup(t, up, srv.URL+"/go/v1")

	postAnthropic(t, router.URL, "/v1/messages",
		mustJSON(searchToolRequest(t, router.URL, nil)))

	up.mu.Lock()
	defer up.mu.Unlock()
	if len(up.requests) < 2 {
		t.Fatalf("only %d upstream round(s); the loop did not run", len(up.requests))
	}
	second, _ := json.Marshal(up.requests[1])
	body := string(second)
	if !strings.Contains(body, `"role":"tool"`) {
		t.Fatalf("no tool message in the second round: %s", body)
	}
	if !strings.Contains(body, "February 2027") {
		t.Errorf("the snippet never reached the model: %s", body)
	}
}

// A Claude model id has to resolve, or a Claude Code session dies before it starts.
func TestClaudeModelIDsResolve(t *testing.T) {
	up := &scriptedUpstream{replies: []string{answeredReply}}
	srv := up.serve(t, "")
	t.Cleanup(srv.Close)
	router := serverToolSetup(t, up, srv.URL+"/go/v1")

	for _, id := range []string{
		"claude-sonnet-4-5-20250929", "claude-haiku-4-5", "claude-3-5-haiku-20241022",
		"claude-opus-5-5", "claude-3.7-sonnet", "claude",
	} {
		_, raw := postAnthropic(t, router.URL, "/v1/messages", mustJSON(map[string]any{
			"model": id, "max_tokens": 16,
			"messages": []any{map[string]any{"role": "user", "content": "hi"}}}))
		if strings.Contains(string(raw), "unknown model") {
			t.Errorf("%s did not resolve: %s", id, raw)
			continue
		}
		// The answer must echo the model the client asked for, or the client thinks it was
		// rerouted somewhere else.
		if !strings.Contains(string(raw), `"model":"`+id+`"`) {
			t.Errorf("response model is not echoed for %s: %s", id, raw)
		}
	}
	// A genuine typo must still fail: aliasing must not swallow every unknown name.
	_, raw := postAnthropic(t, router.URL, "/v1/messages", mustJSON(map[string]any{
		"model": "gemma4-nope", "max_tokens": 16,
		"messages": []any{map[string]any{"role": "user", "content": "hi"}}}))
	if !strings.Contains(string(raw), "unknown model") {
		t.Errorf("a non-Claude unknown id must still 400: %s", raw)
	}
}

// An unsupported server tool type must fail with a name, not be quietly dropped: the
// client would otherwise believe the capability is available.
func TestUnsupportedServerToolIsNamed(t *testing.T) {
	up := &scriptedUpstream{replies: []string{answeredReply}}
	srv := up.serve(t, "")
	t.Cleanup(srv.Close)
	router := serverToolSetup(t, up, srv.URL+"/go/v1")

	resp, raw := postAnthropic(t, router.URL, "/v1/messages", mustJSON(map[string]any{
		"model": canonicalModel, "max_tokens": 16,
		"tools":    []any{map[string]any{"type": "code_execution_20250825", "name": "code_execution"}},
		"messages": []any{map[string]any{"role": "user", "content": "hi"}}}))
	if resp.StatusCode != 400 {
		t.Fatalf("status = %d, want 400: %s", resp.StatusCode, raw)
	}
	if !strings.Contains(string(raw), "code_execution") {
		t.Errorf("the refused type is not named: %s", raw)
	}
}

// A streamed request with server tools must show the search blocks live and then the
// answer, in index order, ending with message_stop.
func TestServerToolStreaming(t *testing.T) {
	roundOne := `data: {"choices":[{"delta":{"content":"Searching."}}]}` + "\n\n" +
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"web_search","arguments":"{\"query\":\"go 1.27\"}"}}]}}]}` + "\n\n" +
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" +
		`data: {"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":9}}` + "\n\n" +
		"data: [DONE]\n\n"
	roundTwo := `data: {"choices":[{"delta":{"content":"Go 1.27 lands in February 2027."}}]}` + "\n\n" +
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
		`data: {"choices":[],"usage":{"prompt_tokens":300,"completion_tokens":15}}` + "\n\n" +
		"data: [DONE]\n\n"

	up := &scriptedUpstream{stream: []string{roundOne, roundTwo}}
	srv := up.serve(t, "")
	t.Cleanup(srv.Close)
	router := serverToolSetup(t, up, srv.URL+"/go/v1")

	req := searchToolRequest(t, router.URL, map[string]any{"stream": true})
	resp, raw := postAnthropic(t, router.URL, "/v1/messages", mustJSON(req))
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d: %s", resp.StatusCode, raw)
	}

	events := parseSSE(t, raw)
	var names []string
	var text strings.Builder
	var indexes []int
	sawSearchResult := false
	for _, ev := range events {
		names = append(names, ev.Name)
		switch ev.Name {
		case "content_block_start":
			if idx, ok := ev.Data["index"].(float64); ok {
				indexes = append(indexes, int(idx))
			}
			if blk, ok := ev.Data["content_block"].(map[string]any); ok &&
				blk["type"] == "web_search_tool_result" {
				sawSearchResult = true
			}
		case "content_block_delta":
			if d, ok := ev.Data["delta"].(map[string]any); ok && d["type"] == "text_delta" {
				text.WriteString(d["text"].(string))
			}
		}
	}

	if names[0] != "message_start" || names[len(names)-1] != "message_stop" {
		t.Fatalf("stream did not open and close properly: %v", names)
	}
	if !sawSearchResult {
		t.Errorf("no web_search_tool_result block in the stream: %v", names)
	}
	if !strings.Contains(text.String(), "February 2027") {
		t.Errorf("streamed text = %q, want the answer from the second round", text.String())
	}
	// Block indexes are what a client keys its content array on: they must be unique and
	// keep increasing across rounds, so the searches render before the answer.
	if len(indexes) < 4 {
		t.Fatalf("too few blocks for two rounds: %v", indexes)
	}
	for i := 1; i < len(indexes); i++ {
		if indexes[i] <= indexes[i-1] {
			t.Fatalf("block indexes are not increasing: %v", indexes)
		}
	}
}

// History sent back by the client must survive translation: the model has to see what it
// read on the previous turn.
func TestServerToolHistoryRoundTrip(t *testing.T) {
	up := &scriptedUpstream{replies: []string{answeredReply}}
	srv := up.serve(t, "")
	t.Cleanup(srv.Close)
	router := serverToolSetup(t, up, srv.URL+"/go/v1")

	sealed := sealSnippet("https://go.dev/doc/devel/release.html", "Go Release Notes",
		"Go 1.27 is planned for February 2027.")
	req := map[string]any{
		"model": canonicalModel, "max_tokens": 64,
		"messages": []any{
			map[string]any{"role": "user", "content": "When is Go 1.27 out?"},
			map[string]any{"role": "assistant", "content": []any{
				map[string]any{"type": "text", "text": "I'll look that up."},
				map[string]any{"type": "server_tool_use", "id": "srvtoolu_1",
					"name": "web_search", "input": map[string]any{"query": "go 1.27"}},
				map[string]any{"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
					"content": []any{map[string]any{
						"type": "web_search_result", "url": "https://go.dev/doc/devel/release.html",
						"title": "Go Release Notes", "encrypted_content": sealed}},
				},
			}},
			map[string]any{"role": "user", "content": "So which month?"},
		},
	}
	resp, raw := postAnthropic(t, router.URL, "/v1/messages", mustJSON(req))
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d: %s", resp.StatusCode, raw)
	}

	up.mu.Lock()
	defer up.mu.Unlock()
	body, _ := json.Marshal(up.requests[0])
	if !strings.Contains(string(body), "February 2027") {
		t.Fatalf("the earlier snippet was dropped from the conversation: %s", body)
	}
	if !strings.Contains(string(body), `"role":"tool"`) {
		t.Errorf("the earlier search result is not a tool message: %s", body)
	}
	// The tool message is only meaningful if the assistant turn it answers was translated
	// too: without the matching tool_calls the upstream sees an orphan result, and providers
	// reject that (or ignore it).
	if !strings.Contains(string(body), `"tool_calls"`) ||
		!strings.Contains(string(body), `"name":"web_search"`) {
		t.Errorf("the server_tool_use block did not become an upstream tool_call: %s", body)
	}
	// The text the model wrote before searching must survive as well.
	if !strings.Contains(string(body), "I'll look that up.") {
		t.Errorf("the assistant text was dropped: %s", body)
	}
}
