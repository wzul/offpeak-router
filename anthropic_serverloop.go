package main

// The server-tool loop.
//
// One client request can become several upstream calls: the model asks for a search,
// this side runs it, the model reads the results and either searches again or answers.
// The client sees a single Anthropic response whose content blocks record what
// happened, exactly as if the API had run the tools itself.

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// errAlreadyWritten marks a failure that happened after the response body started.
// Failover is impossible at that point: a second provider would append a second message
// to a stream the client is already reading.
var errAlreadyWritten = errors.New("response already started")

type alreadyWritten struct{ err error }

func (e alreadyWritten) Error() string { return e.err.Error() }
func (e alreadyWritten) Unwrap() error { return e.err }

// serverLoopState is what the loop learns about a request as it runs.
type serverLoopState struct {
	name        string
	model       string
	clientModel string
	servers     []serverToolSpec
	usage       usageBlock
	searches    int
	fetches     int
	now         time.Time
}

// serveAnthropicServerTools runs the loop and writes the Anthropic response.
func serveAnthropicServerTools(w http.ResponseWriter, r *http.Request, name string, p *Provider,
	key string, payload map[string]any, req *anthropicRequest, servers []serverToolSpec,
	stream bool, now time.Time, model, clientModel string) error {

	st := &serverLoopState{name: name, model: model, clientModel: clientModel,
		servers: servers, now: now}

	base := make(map[string]any, len(payload)+3)
	for k, v := range payload {
		base[k] = v
	}
	base["model"] = upstreamModel(name, model)
	base["stream"] = stream
	if stream {
		base["stream_options"] = map[string]any{"include_usage": true}
	}

	// Conversation state: what the model has already seen and said.
	msgs, _ := base["messages"].([]any)
	messages := append([]any{}, msgs...)

	// Headers first: a stream sink writes the status line immediately, and a header set
	// afterwards is silently dropped.
	w.Header().Set("X-Router-Provider", name)
	w.Header().Set("X-Router-Model", model)

	var out outSink
	if stream {
		out = newStreamSink(w, clientModel)
	} else {
		out = newBufferSink(clientModel)
	}

	for round := 0; round < maxServerToolRounds; round++ {
		body := make(map[string]any, len(base)+1)
		for k, v := range base {
			body[k] = v
		}
		body["messages"] = messages

		resp, err := postRound(r, p, key, body)
		if err != nil {
			if out.started() {
				out.fail(err.Error())
				return alreadyWritten{err}
			}
			return err
		}

		var roundText strings.Builder
		var calls []upstreamToolCall
		var roundUsage *usageBlock
		var finish string

		if stream {
			roundUsage, calls, finish, err = readRoundStream(resp.Body, out, &roundText)
		} else {
			roundUsage, calls, finish, err = readRoundOnce(resp.Body, out, &roundText)
		}
		resp.Body.Close()
		if err != nil {
			if out.started() {
				out.fail(err.Error())
				return alreadyWritten{err}
			}
			return err
		}

		if roundUsage != nil {
			st.usage.PromptTokens += roundUsage.PromptTokens
			st.usage.CompletionTokens += roundUsage.CompletionTokens
			st.usage.PromptTokensDetails.CachedTokens += roundUsage.PromptTokensDetails.CachedTokens
			st.usage.PromptCacheHitTokens += roundUsage.PromptCacheHitTokens
		}
		record(name, model, roundUsage, now)

		// Split the model's tool calls: the ones we execute, and the ones the client does.
		serverCalls, clientCalls := partitionToolCalls(calls)

		if len(serverCalls) == 0 && len(clientCalls) == 0 {
			out.closeOpen()
			reason := anthropicStopReason(finish, false)
			out.finish(reason, st.usageMap())
			log.Printf("anthropic/messages model=%s provider=%s rounds=%d searches=%d fetches=%d",
				model, name, round+1, st.searches, st.fetches)
			return out.flush(w)
		}

		if len(clientCalls) > 0 {
			// The client owns these. Anything we executed this round is already in front
			// of them; the client runs its tools and comes back with the results.
			for _, tc := range clientCalls {
				out.clientToolUse(tc)
			}
			out.closeOpen()
			out.finish("tool_use", st.usageMap())
			log.Printf("anthropic/messages model=%s provider=%s rounds=%d stop=tool_use",
				model, name, round+1)
			return out.flush(w)
		}

		// Only server tools were requested: run them here and keep the loop going.
		executed := make([]serverToolCall, 0, len(serverCalls))
		for _, tc := range serverCalls {
			input := decodeToolInput(tc.Function.Arguments)
			call := executeServerTool(r.Context(), st.servers, tc.ID, tc.Function.Name, input)
			if call.Kind == serverToolSearch && call.Error == "" {
				st.searches++
			}
			if call.Kind == serverToolFetch && call.Error == "" {
				st.fetches++
			}
			executed = append(executed, call)
		}

		out.closeOpen()
		for _, call := range executed {
			out.serverCall(call)
		}

		// Feed the model its own call plus the results, in the shape it produced.
		assistant := map[string]any{"role": "assistant", "content": roundText.String()}
		rawCalls := make([]any, 0, len(serverCalls))
		for _, tc := range serverCalls {
			rawCalls = append(rawCalls, map[string]any{
				"id": tc.ID, "type": "function",
				"function": map[string]any{
					"name": tc.Function.Name, "arguments": defaultArgs(tc.Function.Arguments)},
			})
		}
		assistant["tool_calls"] = rawCalls
		messages = append(messages, assistant)
		for _, call := range executed {
			messages = append(messages, call.toolMessage())
		}
	}

	// Rounds exhausted: answer with what has been gathered rather than hanging.
	out.closeOpen()
	out.finish("end_turn", st.usageMap())
	log.Printf("anthropic/messages model=%s provider=%s rounds=max searches=%d fetches=%d",
		model, name, st.searches, st.fetches)
	return out.flush(w)
}

func defaultArgs(args string) string {
	if strings.TrimSpace(args) == "" {
		return "{}"
	}
	return args
}

// partitionToolCalls separates our synthetic server tools from the client's functions.
func partitionToolCalls(calls []upstreamToolCall) (server, client []upstreamToolCall) {
	names := serverToolNames()
	for _, tc := range calls {
		if _, ok := names[tc.Function.Name]; ok {
			server = append(server, tc)
			continue
		}
		client = append(client, tc)
	}
	return server, client
}

func decodeToolInput(args string) map[string]any {
	out := map[string]any{}
	if strings.TrimSpace(args) == "" {
		return out
	}
	if json.Unmarshal([]byte(args), &out) != nil {
		return map[string]any{}
	}
	return out
}

// postRound issues one upstream call with the router's own headers and auth.
func postRound(client *http.Request, p *Provider, key string, body map[string]any) (*http.Response, error) {
	buf, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("encode body: %w", err)
	}
	req, err := http.NewRequestWithContext(client.Context(), http.MethodPost,
		p.Base+"/chat/completions", strings.NewReader(string(buf)))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	applyUpstreamHeaders(req, p, client, body)

	resp, err := httpc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("transport: %w", err)
	}
	if resp.StatusCode >= 400 {
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 400))
		resp.Body.Close()
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(detail)))
	}
	return resp, nil
}

// readRoundOnce consumes a complete upstream answer.
func readRoundOnce(body io.Reader, out outSink, text *strings.Builder) (
	*usageBlock, []upstreamToolCall, string, error) {

	raw, err := io.ReadAll(body)
	if err != nil {
		return nil, nil, "", fmt.Errorf("read body: %w", err)
	}
	var env upstreamChat
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, nil, "", fmt.Errorf("decode upstream: %w", err)
	}
	if len(env.Choices) == 0 {
		return env.Usage, nil, "", nil
	}
	c := env.Choices[0]
	if c.Message.ReasoningContent != "" {
		out.thinking(c.Message.ReasoningContent)
	}
	if c.Message.Content != "" {
		out.text(c.Message.Content)
		text.WriteString(c.Message.Content)
	}
	return env.Usage, c.Message.ToolCalls, c.FinishReason, nil
}

// readRoundStream consumes one streamed round, relaying text as it arrives and
// collecting the tool calls the model asked for.
func readRoundStream(body io.Reader, out outSink, text *strings.Builder) (
	*usageBlock, []upstreamToolCall, string, error) {

	var (
		captured *usageBlock
		finish   string
		ordered  []upstreamToolCall
		byIndex  = map[int]int{} // upstream index -> position in ordered
	)

	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok || strings.TrimSpace(data) == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content          string             `json:"content"`
					ReasoningContent string             `json:"reasoning_content"`
					ToolCalls        []upstreamToolCall `json:"tool_calls"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage *usageBlock `json:"usage"`
		}
		if json.Unmarshal([]byte(data), &chunk) != nil {
			continue
		}
		if chunk.Usage != nil {
			captured = chunk.Usage
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		c := chunk.Choices[0]
		if c.FinishReason != "" {
			finish = c.FinishReason
		}
		if c.Delta.ReasoningContent != "" {
			out.thinking(c.Delta.ReasoningContent)
		}
		if c.Delta.Content != "" {
			out.text(c.Delta.Content)
			text.WriteString(c.Delta.Content)
		}
		for _, tc := range c.Delta.ToolCalls {
			pos, seen := byIndex[tc.Index]
			if !seen {
				pos = len(ordered)
				byIndex[tc.Index] = pos
				id := tc.ID
				if id == "" {
					id = "toolu_" + strings.TrimPrefix(newMessageID(), "msg_")
				}
				ordered = append(ordered, upstreamToolCall{
					Index: tc.Index, ID: id, Type: "function",
					Function: struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					}{Name: tc.Function.Name},
				})
			}
			if tc.ID != "" && ordered[pos].ID == "" {
				ordered[pos].ID = tc.ID
			}
			if tc.Function.Name != "" && ordered[pos].Function.Name == "" {
				ordered[pos].Function.Name = tc.Function.Name
			}
			ordered[pos].Function.Arguments += tc.Function.Arguments
		}
	}
	if err := sc.Err(); err != nil {
		return captured, ordered, finish, fmt.Errorf("read stream: %w", err)
	}
	return captured, ordered, finish, nil
}

// usageMap is the usage envelope Anthropic clients expect, including the server-tool
// counters that tell a client how many searches it was charged for.
func (st *serverLoopState) usageMap() map[string]any {
	usage := map[string]any{
		"input_tokens":  st.usage.PromptTokens,
		"output_tokens": st.usage.CompletionTokens,
	}
	server := map[string]any{}
	if st.searches > 0 {
		server["web_search_requests"] = st.searches
	}
	if st.fetches > 0 {
		server["web_fetch_requests"] = st.fetches
	}
	if len(server) > 0 {
		usage["server_tool_use"] = server
	}
	return usage
}

// outSink is the shared shape of the two renderers: a buffered message and a live
// stream. The loop must not care which one it is writing to.
type outSink interface {
	started() bool
	thinking(text string)
	text(text string)
	closeOpen()
	serverCall(call serverToolCall)
	clientToolUse(tc upstreamToolCall)
	finish(stopReason string, usage map[string]any)
	fail(detail string)
	flush(w http.ResponseWriter) error
}

// bufferSink collects blocks and writes one JSON message at the end.
type bufferSink struct {
	clientModel string
	blocks      []any
	stopReason  string
	usage       map[string]any
	begin       bool
}

func newBufferSink(clientModel string) *bufferSink {
	return &bufferSink{clientModel: clientModel}
}

func (b *bufferSink) started() bool { return b.begin }

func (b *bufferSink) thinking(text string) {
	b.begin = true
	if n := len(b.blocks); n > 0 {
		if last, ok := b.blocks[n-1].(map[string]any); ok && last["type"] == "thinking" {
			last["thinking"] = last["thinking"].(string) + text
			return
		}
	}
	b.blocks = append(b.blocks, map[string]any{
		"type": "thinking", "thinking": text, "signature": ""})
}

func (b *bufferSink) text(text string) {
	b.begin = true
	if n := len(b.blocks); n > 0 {
		if last, ok := b.blocks[n-1].(map[string]any); ok && last["type"] == "text" {
			last["text"] = last["text"].(string) + text
			return
		}
	}
	b.blocks = append(b.blocks, map[string]any{"type": "text", "text": text})
}

func (b *bufferSink) closeOpen() {}

func (b *bufferSink) serverCall(call serverToolCall) {
	b.begin = true
	b.blocks = append(b.blocks, call.blocks()...)
}

func (b *bufferSink) clientToolUse(tc upstreamToolCall) {
	b.begin = true
	id := tc.ID
	if id == "" {
		id = "toolu_" + strings.TrimPrefix(newMessageID(), "msg_")
	}
	input := json.RawMessage(tc.Function.Arguments)
	if len(input) == 0 || !json.Valid(input) {
		input = json.RawMessage("{}")
	}
	b.blocks = append(b.blocks, map[string]any{
		"type": "tool_use", "id": id, "name": tc.Function.Name, "input": input})
}

func (b *bufferSink) finish(stopReason string, usage map[string]any) {
	b.stopReason = stopReason
	b.usage = usage
}

func (b *bufferSink) fail(detail string) {}

func (b *bufferSink) flush(w http.ResponseWriter) error {
	if len(b.blocks) == 0 {
		b.blocks = append(b.blocks, map[string]any{"type": "text", "text": ""})
	}
	usage := b.usage
	if usage == nil {
		usage = map[string]any{"input_tokens": 0, "output_tokens": 0}
	}
	writeJSON(w, 200, map[string]any{
		"id":            newMessageID(),
		"type":          "message",
		"role":          "assistant",
		"model":         b.clientModel,
		"content":       b.blocks,
		"stop_reason":   b.stopReason,
		"stop_sequence": nil,
		"usage":         usage,
	})
	return nil
}

// streamSink writes the block sequence live. Indexes keep increasing across rounds, so
// a client renders the searches and then the answer in the order they happened.
type streamSink struct {
	w           io.Writer
	flusher     http.Flusher
	clientModel string
	nextIndex   int
	thinkingAt  int
	textAt      int
	wrote       bool
	begun       bool
}

func newStreamSink(w http.ResponseWriter, clientModel string) *streamSink {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(200)
	flusher, _ := w.(http.Flusher)

	s := &streamSink{w: w, flusher: flusher, clientModel: clientModel,
		thinkingAt: -1, textAt: -1}
	_ = anthropicSSE(w, flusher, "message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": newMessageID(), "type": "message", "role": "assistant",
			"model": clientModel, "content": []any{},
			"stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": 0, "output_tokens": 0},
		},
	})
	_ = anthropicSSE(w, flusher, "ping", map[string]any{"type": "ping"})
	return s
}

func (s *streamSink) started() bool { return s.begun }

func (s *streamSink) openThinking() int {
	if s.thinkingAt < 0 {
		s.thinkingAt = s.nextIndex
		s.nextIndex++
		_ = anthropicSSE(s.w, s.flusher, "content_block_start", map[string]any{
			"type": "content_block_start", "index": s.thinkingAt,
			"content_block": map[string]any{
				"type": "thinking", "thinking": "", "signature": ""}})
	}
	return s.thinkingAt
}

func (s *streamSink) openText() int {
	if s.textAt < 0 {
		s.textAt = s.nextIndex
		s.nextIndex++
		_ = anthropicSSE(s.w, s.flusher, "content_block_start", map[string]any{
			"type": "content_block_start", "index": s.textAt,
			"content_block": map[string]any{"type": "text", "text": ""}})
	}
	return s.textAt
}

func (s *streamSink) thinking(text string) {
	s.begun = true
	idx := s.openThinking()
	_ = anthropicSSE(s.w, s.flusher, "content_block_delta", map[string]any{
		"type": "content_block_delta", "index": idx,
		"delta": map[string]any{"type": "thinking_delta", "thinking": text}})
	s.wrote = true
}

func (s *streamSink) text(text string) {
	s.begun = true
	if s.thinkingAt >= 0 {
		s.closeAt(s.thinkingAt)
		s.thinkingAt = -1
	}
	idx := s.openText()
	_ = anthropicSSE(s.w, s.flusher, "content_block_delta", map[string]any{
		"type": "content_block_delta", "index": idx,
		"delta": map[string]any{"type": "text_delta", "text": text}})
	s.wrote = true
}

func (s *streamSink) closeAt(idx int) {
	if idx < 0 {
		return
	}
	_ = anthropicSSE(s.w, s.flusher, "content_block_stop",
		map[string]any{"type": "content_block_stop", "index": idx})
}

func (s *streamSink) closeOpen() {
	if s.thinkingAt >= 0 {
		s.closeAt(s.thinkingAt)
		s.thinkingAt = -1
	}
	if s.textAt >= 0 {
		s.closeAt(s.textAt)
		s.textAt = -1
	}
}

// serverCall writes the two blocks a server tool produces, whole: a client cannot act
// on them mid-way, and Anthropic sends them as complete blocks too.
func (s *streamSink) serverCall(call serverToolCall) {
	s.begun = true
	s.wrote = true
	for _, blk := range call.blocks() {
		idx := s.nextIndex
		s.nextIndex++
		_ = anthropicSSE(s.w, s.flusher, "content_block_start", map[string]any{
			"type": "content_block_start", "index": idx, "content_block": blk})
		s.closeAt(idx)
	}
}

func (s *streamSink) clientToolUse(tc upstreamToolCall) {
	s.begun = true
	s.wrote = true
	id := tc.ID
	if id == "" {
		id = "toolu_" + strings.TrimPrefix(newMessageID(), "msg_")
	}
	idx := s.nextIndex
	s.nextIndex++
	_ = anthropicSSE(s.w, s.flusher, "content_block_start", map[string]any{
		"type": "content_block_start", "index": idx,
		"content_block": map[string]any{
			"type": "tool_use", "id": id, "name": tc.Function.Name,
			"input": json.RawMessage("{}")}})
	if args := strings.TrimSpace(tc.Function.Arguments); args != "" {
		_ = anthropicSSE(s.w, s.flusher, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": idx,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": args}})
	}
	s.closeAt(idx)
}

func (s *streamSink) finish(stopReason string, usage map[string]any) {
	if !s.wrote {
		// Never leave a client waiting on a stream that carried nothing.
		_ = anthropicSSE(s.w, s.flusher, "content_block_start", map[string]any{
			"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "text", "text": ""}})
		s.closeAt(0)
	}
	_ = anthropicSSE(s.w, s.flusher, "message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil},
		"usage": usage,
	})
	_ = anthropicSSE(s.w, s.flusher, "message_stop", map[string]any{"type": "message_stop"})
}

// fail reports a mid-stream failure in the one way an Anthropic client understands.
func (s *streamSink) fail(detail string) {
	_ = anthropicSSE(s.w, s.flusher, "error", map[string]any{
		"type": "error",
		"error": map[string]any{
			"type": "api_error", "message": "upstream failed mid-conversation: " + detail},
	})
}

func (s *streamSink) flush(w http.ResponseWriter) error { return nil }
