package main

// The Anthropic dialect is only worth having if a real Anthropic client can parse
// it. These tests pin the wire shapes, not the internals: request translation, the
// reply envelope, the stop_reason vocabulary, and the SSE event grammar - which is
// a sequence of named events, so an OpenAI-style data-only stream is not "close
// enough".

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

// anthropicSetup wires a scripted upstream so each test controls the exact reply.
func anthropicSetup(t *testing.T, handler http.HandlerFunc) (*httptest.Server, *[]map[string]any) {
	t.Helper()
	var seen []map[string]any
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		seen = append(seen, body)
		handler(w, r)
	}))
	t.Cleanup(up.Close)

	t.Setenv("OPENCODE_API_KEY", "k-go")
	origProvs, origModel := provs, canonicalModel
	t.Cleanup(func() { provs = origProvs; canonicalModel = origModel })
	provs = map[string]*Provider{
		"go": {Base: up.URL + "/go/v1", KeyEnv: "OPENCODE_API_KEY",
			Models: map[string]ModelRoute{
				"mock-model": {ID: "mock-go"},
				"clef-model": {ID: "mock-clef", API: APISystemOne},
			}},
	}
	led = NewLedger(filepath.Join(t.TempDir(), "spend.json"))
	breakers = map[string]*breakerEntry{}
	shadow = false
	authToken = ""
	nowFn = func() time.Time { return mon(13) } // off-peak: go is chosen

	srv := httptest.NewServer(newMux())
	t.Cleanup(srv.Close)
	return srv, &seen
}

func postAnthropic(t *testing.T, url, path, body string) (*http.Response, []byte) {
	t.Helper()
	resp, err := http.Post(url+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	b, _ := io.ReadAll(resp.Body)
	return resp, b
}

func plainReply(content string, finish string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"choices":[{"message":{"content":%q},"finish_reason":%q}],
			"usage":{"prompt_tokens":1000,"completion_tokens":100}}`, content, finish)
	}
}

// Every Anthropic field that has an OpenAI home must actually move, or the model
// answers the wrong question.
func TestAnthropicRequestTranslation(t *testing.T) {
	srv, seen := anthropicSetup(t, plainReply("ok", "stop"))

	body := `{
	 "model":"mock-model","max_tokens":256,"temperature":0.3,"top_p":0.9,"top_k":40,
	 "stop_sequences":["END"],"stream":false,
	 "system":[{"type":"text","text":"be brief"}],
	 "tools":[{"name":"lookup","description":"d","input_schema":{"type":"object",
	   "properties":{"q":{"type":"string"}}}}],
	 "tool_choice":{"type":"any"},
	 "messages":[
	   {"role":"user","content":[{"type":"text","text":"hi"},
	     {"type":"image","source":{"type":"base64","media_type":"image/png","data":"QUJD"}}]},
	   {"role":"assistant","content":[{"type":"tool_use","id":"tu1","name":"lookup",
	     "input":{"q":"x"}}]},
	   {"role":"user","content":[{"type":"tool_result","tool_use_id":"tu1",
	     "content":[{"type":"text","text":"result text"}]}]}
	 ]}`
	resp, raw := postAnthropic(t, srv.URL, "/v1/messages", body)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, body = %s", resp.StatusCode, raw)
	}
	if len(*seen) != 1 {
		t.Fatalf("upstream calls = %d, want 1", len(*seen))
	}
	up := (*seen)[0]

	if up["max_tokens"] != float64(256) || up["temperature"] != 0.3 || up["top_p"] != 0.9 {
		t.Errorf("sampling params lost: max_tokens=%v temperature=%v top_p=%v",
			up["max_tokens"], up["temperature"], up["top_p"])
	}
	if _, has := up["top_k"]; has {
		t.Error("top_k has no OpenAI equivalent and must not be forwarded")
	}
	if stops, _ := up["stop"].([]any); len(stops) != 1 || stops[0] != "END" {
		t.Errorf("stop_sequences not mapped to stop: %v", up["stop"])
	}
	if up["model"] != "mock-go" {
		t.Errorf("model = %v, want the provider id mock-go", up["model"])
	}

	msgs, _ := up["messages"].([]any)
	// Four, not five: a user turn carrying only tool_result becomes just the tool
	// message, with no empty user message left behind for the model to answer.
	if len(msgs) != 4 {
		t.Fatalf("messages = %d, want 4 (system, user+image, assistant+tool_calls, tool): %s",
			len(msgs), mustJSON(up["messages"]))
	}
	sys := msgs[0].(map[string]any)
	if sys["role"] != "system" || sys["content"] != "be brief" {
		t.Errorf("system blocks not flattened: %v", sys)
	}
	user := msgs[1].(map[string]any)
	parts, ok := user["content"].([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("image turn must become content parts: %v", user["content"])
	}
	img := parts[1].(map[string]any)
	urlOf := img["image_url"].(map[string]any)["url"].(string)
	if !strings.HasPrefix(urlOf, "data:image/png;base64,") {
		t.Errorf("base64 image not turned into a data URL: %s", urlOf)
	}
	asst := msgs[2].(map[string]any)
	calls, _ := asst["tool_calls"].([]any)
	if len(calls) != 1 {
		t.Fatalf("tool_use must become tool_calls: %v", asst)
	}
	fn := calls[0].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "lookup" || fn["arguments"] != `{"q":"x"}` {
		t.Errorf("tool_call = %v", fn)
	}
	tool := msgs[3].(map[string]any)
	if tool["role"] != "tool" || tool["tool_call_id"] != "tu1" || tool["content"] != "result text" {
		t.Errorf("tool_result must become a tool message: %v", tool)
	}

	tools, _ := up["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %v", up["tools"])
	}
	if got := tools[0].(map[string]any)["type"]; got != "function" {
		t.Errorf("tool type = %v, want function", got)
	}
	if up["tool_choice"] != "required" {
		t.Errorf("tool_choice any = %v, want required", up["tool_choice"])
	}
}

func mustJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func TestAnthropicResponseEnvelope(t *testing.T) {
	srv, _ := anthropicSetup(t, plainReply("hello", "stop"))
	resp, raw := postAnthropic(t, srv.URL, "/v1/messages",
		`{"model":"mock-model","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, body = %s", resp.StatusCode, raw)
	}

	var out struct {
		ID      string `json:"id"`
		Type    string `json:"type"`
		Role    string `json:"role"`
		Model   string `json:"model"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      struct {
			Input  int `json:"input_tokens"`
			Output int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("decode: %v (%s)", err, raw)
	}
	if !strings.HasPrefix(out.ID, "msg_") {
		t.Errorf("id = %q, want an msg_ id", out.ID)
	}
	if out.Type != "message" || out.Role != "assistant" {
		t.Errorf("envelope = %s/%s, want message/assistant", out.Type, out.Role)
	}
	if out.Model != "mock-model" {
		t.Errorf("model = %q, want the name the client asked for", out.Model)
	}
	if len(out.Content) != 1 || out.Content[0].Type != "text" || out.Content[0].Text != "hello" {
		t.Errorf("content = %+v", out.Content)
	}
	if out.StopReason != "end_turn" {
		t.Errorf("stop_reason = %q, want end_turn", out.StopReason)
	}
	if out.Usage.Input != 1000 || out.Usage.Output != 100 {
		t.Errorf("usage = %+v, want 1000/100", out.Usage)
	}
}

// Anthropic clients switch on stop_reason; the OpenAI vocabulary in that field
// makes them treat a truncated reply as complete.
func TestAnthropicStopReasonVocabulary(t *testing.T) {
	cases := []struct{ finish, want string }{
		{"stop", "end_turn"},
		{"length", "max_tokens"},
		{"tool_calls", "tool_use"},
	}
	for _, c := range cases {
		srv, _ := anthropicSetup(t, plainReply("x", c.finish))
		_, raw := postAnthropic(t, srv.URL, "/v1/messages",
			`{"model":"mock-model","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`)
		var out map[string]any
		_ = json.Unmarshal(raw, &out)
		if out["stop_reason"] != c.want {
			t.Errorf("finish_reason %q -> stop_reason %v, want %q", c.finish, out["stop_reason"], c.want)
		}
	}
}

func TestAnthropicToolUseReply(t *testing.T) {
	srv, _ := anthropicSetup(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"choices":[{"message":{"content":null,"tool_calls":[
			{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}]},
			"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`)
	})
	_, raw := postAnthropic(t, srv.URL, "/v1/messages",
		`{"model":"mock-model","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`)

	var out struct {
		Content []struct {
			Type  string          `json:"type"`
			ID    string          `json:"id"`
			Name  string          `json:"name"`
			Input json.RawMessage `json:"input"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("decode: %v (%s)", err, raw)
	}
	if out.StopReason != "tool_use" {
		t.Errorf("stop_reason = %q, want tool_use", out.StopReason)
	}
	var tool map[string]any
	for _, c := range out.Content {
		if c.Type == "tool_use" {
			if c.ID != "call_1" || c.Name != "lookup" {
				t.Errorf("tool_use block = %+v", c)
			}
			if err := json.Unmarshal(c.Input, &tool); err != nil || tool["q"] != "x" {
				t.Errorf("tool input must stay an object, got %s", c.Input)
			}
		}
	}
	if tool == nil {
		t.Fatalf("no tool_use block in %s", raw)
	}
}

// A thinking model with a tight budget spends it all on reasoning. Dropping
// reasoning_content leaves the client with an empty message and no explanation.
func TestAnthropicThinkingBlock(t *testing.T) {
	srv, _ := anthropicSetup(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"choices":[{"message":{"content":"Blue",
			"reasoning_content":"sky is blue"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":10,"completion_tokens":5}}`)
	})
	_, raw := postAnthropic(t, srv.URL, "/v1/messages",
		`{"model":"mock-model","max_tokens":64,"messages":[{"role":"user","content":"colour?"}]}`)

	var out struct {
		Content []struct {
			Type     string `json:"type"`
			Text     string `json:"text"`
			Thinking string `json:"thinking"`
		} `json:"content"`
	}
	_ = json.Unmarshal(raw, &out)
	if len(out.Content) < 2 {
		t.Fatalf("content = %s, want a thinking block before the text", raw)
	}
	if out.Content[0].Type != "thinking" || out.Content[0].Thinking != "sky is blue" {
		t.Errorf("first block = %+v, want thinking", out.Content[0])
	}
	if out.Content[1].Type != "text" || out.Content[1].Text != "Blue" {
		t.Errorf("second block = %+v, want text Blue", out.Content[1])
	}
}

type sseEvent struct {
	Name string
	Data map[string]any
}

func parseSSE(t *testing.T, raw []byte) []sseEvent {
	t.Helper()
	var events []sseEvent
	for _, block := range strings.Split(string(raw), "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		var ev sseEvent
		for _, line := range strings.Split(block, "\n") {
			if name, ok := strings.CutPrefix(line, "event: "); ok {
				ev.Name = name
			}
			if payload, ok := strings.CutPrefix(line, "data: "); ok {
				if err := json.Unmarshal([]byte(payload), &ev.Data); err != nil {
					t.Fatalf("bad SSE payload %q: %v", payload, err)
				}
			}
		}
		events = append(events, ev)
	}
	return events
}

func TestAnthropicStreamGrammar(t *testing.T) {
	srv, _ := anthropicSetup(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hmm\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"He\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	})
	resp, raw := postAnthropic(t, srv.URL, "/v1/messages",
		`{"model":"mock-model","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}

	events := parseSSE(t, raw)
	if len(events) < 6 {
		t.Fatalf("events = %d, want a full grammar: %s", len(events), raw)
	}
	if events[0].Name != "message_start" {
		t.Errorf("first event = %q, want message_start", events[0].Name)
	}
	// Thinking must open and close before the text block starts: clients render
	// blocks in index order, and an interleaved pair renders out of order.
	var order []string
	for _, ev := range events {
		if ev.Data["type"] == nil {
			continue
		}
		idx, _ := ev.Data["index"].(float64)
		order = append(order, fmt.Sprintf("%s/%d", ev.Name, int(idx)))
	}
	joined := strings.Join(order, " ")
	if !strings.Contains(joined, "content_block_start/0 content_block_delta/0 content_block_stop/0 content_block_start/1") {
		t.Errorf("block sequence = %s, want thinking block 0 closed before text block 1", joined)
	}
	if last := events[len(events)-1].Name; last != "message_stop" {
		t.Errorf("last event = %q, want message_stop", last)
	}

	var deltas []string
	var stopReason string
	var outTokens float64
	for _, ev := range events {
		switch ev.Name {
		case "content_block_delta":
			if d, ok := ev.Data["delta"].(map[string]any); ok && d["type"] == "text_delta" {
				deltas = append(deltas, d["text"].(string))
			}
		case "message_delta":
			if d, ok := ev.Data["delta"].(map[string]any); ok {
				stopReason, _ = d["stop_reason"].(string)
			}
			if u, ok := ev.Data["usage"].(map[string]any); ok {
				outTokens, _ = u["output_tokens"].(float64)
			}
		}
	}
	if strings.Join(deltas, "") != "Hello" {
		t.Errorf("text deltas = %v, want Hello", deltas)
	}
	if stopReason != "end_turn" {
		t.Errorf("message_delta stop_reason = %q, want end_turn", stopReason)
	}
	if int(outTokens) != 3 {
		t.Errorf("usage.output_tokens = %v, want 3", outTokens)
	}
}

func TestAnthropicStreamToolUse(t *testing.T) {
	srv, _ := anthropicSetup(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_9\","+
			"\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\":\"}}]}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,"+
			"\"function\":{\"arguments\":\"\\\"x\\\"}\"}}]}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	})
	_, raw := postAnthropic(t, srv.URL, "/v1/messages",
		`{"model":"mock-model","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	events := parseSSE(t, raw)

	var partial, blockType string
	var stopReason string
	for _, ev := range events {
		switch ev.Name {
		case "content_block_start":
			if cb, ok := ev.Data["content_block"].(map[string]any); ok {
				blockType, _ = cb["type"].(string)
				if blockType == "tool_use" && cb["id"] != "call_9" {
					t.Errorf("tool_use start id = %v", cb["id"])
				}
			}
		case "content_block_delta":
			if d, ok := ev.Data["delta"].(map[string]any); ok && d["type"] == "input_json_delta" {
				partial += d["partial_json"].(string)
			}
		case "message_delta":
			if d, ok := ev.Data["delta"].(map[string]any); ok {
				stopReason, _ = d["stop_reason"].(string)
			}
		}
	}
	if blockType != "tool_use" {
		t.Errorf("content_block type = %q, want tool_use", blockType)
	}
	if partial != `{"q":"x"}` {
		t.Errorf("partial_json = %q, want the arguments reassembled", partial)
	}
	if stopReason != "tool_use" {
		t.Errorf("stop_reason = %q, want tool_use", stopReason)
	}
}

// Errors must arrive as Anthropic's typed envelope; a bare {"error":"..."} makes
// clients report a transport fault instead of the real cause.
func TestAnthropicErrorEnvelope(t *testing.T) {
	srv, _ := anthropicSetup(t, plainReply("ok", "stop"))
	resp, raw := postAnthropic(t, srv.URL, "/v1/messages",
		`{"model":"nope","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 400 {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
	var out struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("decode: %v (%s)", err, raw)
	}
	if out.Type != "error" || out.Error.Type != "invalid_request_error" {
		t.Errorf("error envelope = %+v", out)
	}
	if !strings.Contains(out.Error.Message, "mock-model") {
		t.Errorf("message should list what is available, got %q", out.Error.Message)
	}
}

func TestAnthropicDecisionModelRefused(t *testing.T) {
	srv, _ := anthropicSetup(t, plainReply("ok", "stop"))
	resp, raw := postAnthropic(t, srv.URL, "/v1/messages",
		`{"model":"clef-model","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 400 {
		t.Fatalf("status = %d, want 400: %s", resp.StatusCode, raw)
	}
	if !strings.Contains(string(raw), "systemone") {
		t.Errorf("refusal must point at the route that does serve it: %s", raw)
	}
}

func TestAnthropicRouteRequiresAuth(t *testing.T) {
	srv, _ := anthropicSetup(t, plainReply("ok", "stop"))
	authToken = "secret"
	for _, path := range []string{"/v1/messages", "/v1/messages/count_tokens"} {
		resp, _ := postAnthropic(t, srv.URL, path,
			`{"model":"mock-model","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
		if resp.StatusCode != 401 {
			t.Errorf("%s without a token = %d, want 401", path, resp.StatusCode)
		}
	}
}

// count_tokens must be measured, not guessed: the upstream's own prompt_tokens is
// the only honest number available without a tokenizer.
func TestAnthropicCountTokens(t *testing.T) {
	srv, seen := anthropicSetup(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"choices":[{"message":{"content":""}}],"usage":{"prompt_tokens":4321,"completion_tokens":1}}`)
	})
	resp, raw := postAnthropic(t, srv.URL, "/v1/messages/count_tokens",
		`{"model":"mock-model","messages":[{"role":"user","content":"count me"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, body = %s", resp.StatusCode, raw)
	}
	var out struct {
		Input int `json:"input_tokens"`
	}
	if err := json.Unmarshal(raw, &out); err != nil || out.Input != 4321 {
		t.Fatalf("input_tokens = %d (%s), want the upstream's count", out.Input, raw)
	}
	if len(*seen) == 0 || (*seen)[0]["max_tokens"] != float64(1) {
		t.Errorf("count_tokens must ask for a one-token completion, sent %v", (*seen)[0]["max_tokens"])
	}
}

// Failover must be invisible in the dialect: the client still gets an Anthropic
// message, not a provider error.
func TestAnthropicFailoverKeepsDialect(t *testing.T) {
	srv, _ := anthropicSetup(t, func(w http.ResponseWriter, r *http.Request) {
		// The scripted upstream tells the two providers apart by path.
		if strings.HasPrefix(r.URL.Path, "/go/") {
			w.WriteHeader(500)
			fmt.Fprint(w, `{"error":"boom"}`)
			return
		}
		plainReply("second", "stop")(w, r)
	})
	// A second provider is needed for the chain to have somewhere to go; it points
	// at the same scripted upstream under a different path.
	provs["backup"] = &Provider{Base: strings.Replace(provs["go"].Base, "/go/v1", "/backup/v1", 1),
		KeyEnv: "OPENCODE_API_KEY",
		Models: map[string]ModelRoute{"mock-model": {ID: "mock-backup"}}}

	resp, raw := postAnthropic(t, srv.URL, "/v1/messages",
		`{"model":"mock-model","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, body = %s", resp.StatusCode, raw)
	}
	if resp.Header.Get("X-Router-Failover") != "true" {
		t.Error("X-Router-Failover must be set when the first provider failed")
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("decode: %v (%s)", err, raw)
	}
	if out["type"] != "message" {
		t.Errorf("failover answer = %s, want an Anthropic message", raw)
	}
}
