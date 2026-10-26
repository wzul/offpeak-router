package main

// Anthropic Messages API dialect: POST /v1/messages.
//
// Clients that speak only the Anthropic wire - Claude Code via ANTHROPIC_BASE_URL,
// the @anthropic-ai/sdk, Cursor in Anthropic mode - cannot use the router at all
// without this route. The upstream providers speak OpenAI, so the request is
// translated in and the reply translated out, including the SSE event grammar,
// which is a different sequence of named events, not renamed fields.
//
// Deliberate limits, so nothing is silently invented:
//   - top_k has no OpenAI equivalent and is dropped.
//   - stop_sequences maps to OpenAI stop; stop_reason is reported from the
//     upstream finish_reason, so a stop-sequence hit is reported as end_turn
//     unless the upstream says otherwise.
//   - a reasoning model's reasoning_content becomes a thinking block with an
//     empty signature: the router cannot sign it, and dropping it would make a
//     thinking model look mute when its whole budget went to reasoning.

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

type anthropicSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
	URL       string `json:"url"`
}

type anthropicBlock struct {
	Type      string           `json:"type"`
	Text      string           `json:"text,omitempty"`
	Thinking  string           `json:"thinking,omitempty"`
	Signature string           `json:"signature,omitempty"`
	ID        string           `json:"id,omitempty"`
	Name      string           `json:"name,omitempty"`
	Input     json.RawMessage  `json:"input,omitempty"`
	ToolUseID string           `json:"tool_use_id,omitempty"`
	Content   json.RawMessage  `json:"content,omitempty"`
	Source    *anthropicSource `json:"source,omitempty"`
	IsError   bool             `json:"is_error,omitempty"`
}

type anthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
	// Server tools: the type carries the version (web_search_20250305), and the client
	// expects this side to execute them. A custom function tool has no type or "custom".
	Type           string   `json:"type,omitempty"`
	MaxUses        int      `json:"max_uses,omitempty"`
	AllowedDomains []string `json:"allowed_domains,omitempty"`
	BlockedDomains []string `json:"blocked_domains,omitempty"`
}

type anthropicToolChoice struct {
	Type                   string `json:"type"`
	Name                   string `json:"name"`
	DisableParallelToolUse bool   `json:"disable_parallel_tool_use"`
}

type anthropicRequest struct {
	Model         string               `json:"model"`
	Messages      []anthropicMessage   `json:"messages"`
	System        json.RawMessage      `json:"system"`
	MaxTokens     int                  `json:"max_tokens"`
	Temperature   *float64             `json:"temperature"`
	TopP          *float64             `json:"top_p"`
	TopK          *int                 `json:"top_k"`
	StopSequences []string             `json:"stop_sequences"`
	Stream        bool                 `json:"stream"`
	Tools         []anthropicTool      `json:"tools"`
	ToolChoice    *anthropicToolChoice `json:"tool_choice"`
}

// blocksOrString decodes a field that may be a bare string or an array of blocks.
func blocksOrString(raw json.RawMessage) ([]anthropicBlock, string, error) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return nil, "", nil
	}
	if strings.HasPrefix(trimmed, `"`) {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, "", err
		}
		return nil, s, nil
	}
	var blocks []anthropicBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, "", err
	}
	return blocks, "", nil
}

func textOf(raw json.RawMessage) string {
	blocks, plain, err := blocksOrString(raw)
	if err != nil {
		return ""
	}
	if plain != "" {
		return plain
	}
	var b strings.Builder
	for _, blk := range blocks {
		if blk.Type == "text" {
			b.WriteString(blk.Text)
		}
	}
	return b.String()
}

// toOpenAIChat rewrites an Anthropic request in the shape the providers accept.
func toOpenAIChat(req *anthropicRequest) (map[string]any, error) {
	msgs := make([]any, 0, len(req.Messages)+1)

	if sys := textOf(req.System); strings.TrimSpace(sys) != "" {
		msgs = append(msgs, map[string]any{"role": "system", "content": sys})
	}

	for _, m := range req.Messages {
		blocks, plain, err := blocksOrString(m.Content)
		if err != nil {
			return nil, fmt.Errorf("decode content of %s message: %w", m.Role, err)
		}
		role := m.Role
		if role == "" {
			role = "user"
		}
		if plain != "" || len(blocks) == 0 {
			msgs = append(msgs, map[string]any{"role": role, "content": plain})
			continue
		}

		var texts []string
		var parts []any
		var toolCalls []any
		var toolResults []any

		for _, blk := range blocks {
			switch blk.Type {
			case "text":
				texts = append(texts, blk.Text)
				parts = append(parts, map[string]any{"type": "text", "text": blk.Text})
			case "image":
				if blk.Source == nil {
					continue
				}
				url := blk.Source.URL
				if blk.Source.Type == "base64" && blk.Source.Data != "" {
					url = "data:" + blk.Source.MediaType + ";base64," + blk.Source.Data
				}
				if url == "" {
					continue
				}
				parts = append(parts, map[string]any{
					"type": "image_url", "image_url": map[string]any{"url": url}})
			case "tool_use":
				args := "{}"
				if len(blk.Input) > 0 {
					args = string(blk.Input)
				}
				toolCalls = append(toolCalls, map[string]any{
					"id": blk.ID, "type": "function",
					"function": map[string]any{"name": blk.Name, "arguments": args},
				})
			case "tool_result":
				// Anthropic carries tool results inside a user message; OpenAI
				// wants them as their own messages with role "tool".
				content := textOf(blk.Content)
				if content == "" && len(blk.Content) > 0 {
					content = string(blk.Content)
				}
				toolResults = append(toolResults, map[string]any{
					"role": "tool", "tool_call_id": blk.ToolUseID, "content": content})
			case "thinking":
				// Dropped: the signature cannot be verified here, and forwarding it
				// to an OpenAI upstream is meaningless.
			case "server_tool_use":
				// Our own executed search, sent back on the next turn. The upstream
				// model must see it as the tool call it asked for.
				args := "{}"
				if len(blk.Input) > 0 {
					args = string(blk.Input)
				}
				toolCalls = append(toolCalls, map[string]any{
					"id": blk.ID, "type": "function",
					"function": map[string]any{"name": blk.Name, "arguments": args},
				})
			case "web_search_tool_result", "web_fetch_tool_result":
				// The results the model already read. Losing them would leave the next
				// turn with citations pointing at nothing.
				toolResults = append(toolResults, map[string]any{
					"role": "tool", "tool_call_id": blk.ToolUseID,
					"content": historyText(blk)})
			}
		}

		// Tool results come before the new user text, matching the order Anthropic
		// documents: the tool output answers the previous turn, the text starts the
		// next one.
		msgs = append(msgs, toolResults...)
		if len(toolCalls) > 0 {
			msg := map[string]any{"role": role, "content": strings.Join(texts, "\n")}
			msg["tool_calls"] = toolCalls
			msgs = append(msgs, msg)
			continue
		}
		if len(toolResults) > 0 && len(parts) == 0 {
			continue
		}
		if len(parts) > 1 || hasImagePart(parts) {
			msgs = append(msgs, map[string]any{"role": role, "content": parts})
			continue
		}
		msgs = append(msgs, map[string]any{"role": role, "content": strings.Join(texts, "\n")})
	}

	body := map[string]any{"messages": msgs}
	if req.MaxTokens > 0 {
		body["max_tokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	if len(req.StopSequences) > 0 {
		body["stop"] = req.StopSequences
	}

	if len(req.Tools) > 0 {
		tools := make([]any, 0, len(req.Tools))
		for _, t := range req.Tools {
			fn := map[string]any{"name": t.Name}
			if t.Description != "" {
				fn["description"] = t.Description
			}
			if len(t.InputSchema) > 0 {
				fn["parameters"] = json.RawMessage(t.InputSchema)
			}
			tools = append(tools, map[string]any{"type": "function", "function": fn})
		}
		body["tools"] = tools
		switch {
		case req.ToolChoice == nil || req.ToolChoice.Type == "auto":
			// OpenAI's default is auto; leaving it unset keeps providers with no
			// tool_choice support from rejecting the request.
		case req.ToolChoice.Type == "any":
			body["tool_choice"] = "required"
		case req.ToolChoice.Type == "tool" && req.ToolChoice.Name != "":
			body["tool_choice"] = map[string]any{
				"type": "function", "function": map[string]any{"name": req.ToolChoice.Name}}
		case req.ToolChoice.Type == "none":
			body["tool_choice"] = "none"
		}
	}
	return body, nil
}

func hasImagePart(parts []any) bool {
	for _, p := range parts {
		if m, ok := p.(map[string]any); ok && m["type"] == "image_url" {
			return true
		}
	}
	return false
}

// anthropicStopReason maps an OpenAI finish_reason onto the Anthropic vocabulary.
func anthropicStopReason(finish string, sawTools bool) string {
	switch finish {
	case "length":
		return "max_tokens"
	case "tool_calls", "function_call":
		return "tool_use"
	}
	if sawTools {
		return "tool_use"
	}
	return "end_turn"
}

func newMessageID() string {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		return "msg_router"
	}
	return "msg_" + hex.EncodeToString(b)
}

// anthropicError writes an error in the shape Anthropic clients expect: a typed
// envelope, not a bare message.
func anthropicError(w http.ResponseWriter, code int, kind, message string) {
	writeJSON(w, code, map[string]any{
		"type":  "error",
		"error": map[string]any{"type": kind, "message": message},
	})
}

func handleMessages(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		anthropicError(w, 400, "invalid_request_error", err.Error())
		return
	}
	var req anthropicRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		anthropicError(w, 400, "invalid_request_error", "bad json: "+err.Error())
		return
	}
	if len(req.Messages) == 0 {
		anthropicError(w, 400, "invalid_request_error", "messages is required")
		return
	}

	model, ok := resolveModel(req.Model)
	if !ok {
		anthropicError(w, 400, "invalid_request_error",
			"unknown model: "+req.Model+"; available: "+strings.Join(availableModels(), ", "))
		return
	}
	if api := modelAPI(model); api != APIChat {
		anthropicError(w, 400, "invalid_request_error",
			model+" is a "+api+" model; this route serves chat models only")
		return
	}

	// Server tools are this side's work: the client declares web_search and expects the
	// API to run it. They are split off before translation and turned into functions the
	// upstream model can call, so the loop can execute them inside this request.
	functions, servers, splitErr := splitTools(req.Tools)
	if splitErr != nil {
		anthropicError(w, 400, "invalid_request_error", splitErr.Error())
		return
	}
	declared := req.Tools
	req.Tools = functions
	payload, err := toOpenAIChat(&req)
	req.Tools = declared
	if err != nil {
		anthropicError(w, 400, "invalid_request_error", err.Error())
		return
	}
	if len(servers) > 0 {
		existing, _ := payload["tools"].([]any)
		payload["tools"] = append(existing, syntheticTools(servers)...)
	}

	now := nowFn()
	order := candidateOrder(now, led.Spent(now), model)

	if shadow {
		would := ""
		if len(order) > 0 {
			would = order[0]
		}
		writeJSON(w, 200, map[string]any{"shadow": true, "order": order,
			"would_use": would, "model": model, "dialect": "anthropic/messages"})
		return
	}
	if len(order) == 0 {
		anthropicError(w, 503, "overloaded_error", "no provider can serve "+model+" right now")
		return
	}

	var attempts []map[string]string
	anyFailover := false
	for _, name := range order {
		p := provs[name]
		key, keyErr := providerKey(r.Context(), name, p)
		if keyErr != nil {
			attempts = append(attempts, map[string]string{"provider": name, "error": keyErr.Error()})
			continue
		}
		if breakerOpen(name) {
			attempts = append(attempts, map[string]string{"provider": name, "error": "circuit open"})
			continue
		}
		// Headers are set before anything is written: Set after WriteHeader is
		// silently ignored, and that is exactly when failover happens.
		if anyFailover {
			w.Header().Set("X-Router-Failover", "true")
		}
		var serveErr error
		if len(servers) > 0 {
			serveErr = serveAnthropicServerTools(w, r, name, p, key, payload, &req, servers,
				req.Stream, now, model, req.Model)
		} else {
			serveErr = serveAnthropic(w, r, name, p, key, payload, req.Stream, now, model, req.Model)
		}
		if serveErr == nil {
			breakerOK(name)
			return
		}
		log.Printf("provider %s failed (anthropic/messages): %v", name, serveErr)
		breakerFail(name)
		// A failure after the body started cannot be retried elsewhere: the client is
		// already reading this response.
		var written alreadyWritten
		if errors.As(serveErr, &written) {
			return
		}
		anyFailover = true
		attempts = append(attempts, map[string]string{"provider": name, "error": serveErr.Error()})
	}
	anthropicError(w, 502, "api_error", "all providers failed: "+summariseAttempts(attempts))
}

func summariseAttempts(attempts []map[string]string) string {
	parts := make([]string, 0, len(attempts))
	for _, a := range attempts {
		parts = append(parts, a["provider"]+": "+a["error"])
	}
	return strings.Join(parts, "; ")
}

// serveAnthropic makes one upstream call and writes the answer in Anthropic shape.
// It must not touch w until the upstream answers, or failover becomes impossible.
func serveAnthropic(w http.ResponseWriter, r *http.Request, name string, p *Provider, key string,
	payload map[string]any, stream bool, now time.Time, model, clientModel string) error {

	body := make(map[string]any, len(payload)+3)
	for k, v := range payload {
		body[k] = v
	}
	body["model"] = upstreamModel(name, model)
	body["stream"] = stream
	if stream {
		body["stream_options"] = map[string]any{"include_usage": true}
	}

	buf, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("encode body: %w", err)
	}
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		p.Base+"/chat/completions", bytes.NewReader(buf))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	applyUpstreamHeaders(req, p, r, body)

	resp, err := httpc.Do(req)
	if err != nil {
		return fmt.Errorf("transport: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 400))
		return newUpstreamError(resp.StatusCode, string(detail))
	}

	w.Header().Set("X-Router-Provider", name)
	w.Header().Set("X-Router-Model", model)

	if !stream {
		return renderAnthropicOnce(w, resp.Body, name, model, clientModel, now)
	}
	return renderAnthropicStream(w, resp.Body, name, model, clientModel, now)
}

type upstreamToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type upstreamChat struct {
	Choices []struct {
		Message struct {
			Content          string             `json:"content"`
			ReasoningContent string             `json:"reasoning_content"`
			ToolCalls        []upstreamToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *usageBlock `json:"usage"`
}

func renderAnthropicOnce(w http.ResponseWriter, body io.Reader, name, model, clientModel string,
	now time.Time) error {

	raw, err := io.ReadAll(body)
	if err != nil {
		return fmt.Errorf("read body: %w", err)
	}
	var env upstreamChat
	if err := json.Unmarshal(raw, &env); err != nil {
		return fmt.Errorf("decode upstream: %w", err)
	}
	cost := record(name, model, env.Usage, now)

	out := map[string]any{
		"id":            newMessageID(),
		"type":          "message",
		"role":          "assistant",
		"model":         clientModel,
		"stop_sequence": nil,
	}
	blocks := []any{}
	sawTools := false
	if len(env.Choices) > 0 {
		c := env.Choices[0]
		// Thinking first: it precedes the answer in the model's own output, and an
		// Anthropic client renders blocks in order.
		if c.Message.ReasoningContent != "" {
			blocks = append(blocks, map[string]any{
				"type": "thinking", "thinking": c.Message.ReasoningContent, "signature": ""})
		}
		if c.Message.Content != "" || len(c.Message.ToolCalls) == 0 {
			blocks = append(blocks, map[string]any{"type": "text", "text": c.Message.Content})
		}
		for _, tc := range c.Message.ToolCalls {
			sawTools = true
			input := json.RawMessage("{}")
			if strings.TrimSpace(tc.Function.Arguments) != "" {
				input = json.RawMessage(tc.Function.Arguments)
			}
			id := tc.ID
			if id == "" {
				id = "toolu_" + strings.TrimPrefix(newMessageID(), "msg_")
			}
			blocks = append(blocks, map[string]any{
				"type": "tool_use", "id": id, "name": tc.Function.Name, "input": input})
		}
		out["stop_reason"] = anthropicStopReason(c.FinishReason, sawTools)
	} else {
		out["stop_reason"] = "end_turn"
	}
	out["content"] = blocks

	usage := map[string]any{"input_tokens": 0, "output_tokens": 0}
	if env.Usage != nil {
		usage["input_tokens"] = env.Usage.PromptTokens
		usage["output_tokens"] = env.Usage.CompletionTokens
	}
	out["usage"] = usage

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Router-Usd", fmt.Sprintf("%.6f", cost))
	writeJSON(w, 200, out)
	return nil
}

// anthropicSSE writes one named event. Anthropic clients parse the event name and
// the JSON independently, so both lines are mandatory.
func anthropicSSE(w io.Writer, flusher http.Flusher, event string, payload any) error {
	buf, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, buf); err != nil {
		return err
	}
	if flusher != nil {
		flusher.Flush()
	}
	return nil
}

func renderAnthropicStream(w http.ResponseWriter, body io.Reader, name, model, clientModel string,
	now time.Time) error {

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(200)
	flusher, _ := w.(http.Flusher)

	msgID := newMessageID()
	if err := anthropicSSE(w, flusher, "message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": msgID, "type": "message", "role": "assistant", "model": clientModel,
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			// The upstream reports prompt tokens only with the final chunk, so the
			// real input count lands in message_delta instead of lying here.
			"usage": map[string]any{"input_tokens": 0, "output_tokens": 0},
		},
	}); err != nil {
		return nil
	}
	_ = anthropicSSE(w, flusher, "ping", map[string]any{"type": "ping"})

	var (
		captured   *usageBlock
		finish     string
		sawTools   bool
		nextIndex  int
		thinkingAt = -1
		textAt     = -1
		toolAt     = map[int]int{} // upstream tool index -> content block index
		wrote      bool
	)

	closeBlock := func(idx int) {
		if idx < 0 {
			return
		}
		_ = anthropicSSE(w, flusher, "content_block_stop",
			map[string]any{"type": "content_block_stop", "index": idx})
	}

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
		d := c.Delta

		if d.ReasoningContent != "" {
			if thinkingAt < 0 {
				thinkingAt = nextIndex
				nextIndex++
				_ = anthropicSSE(w, flusher, "content_block_start", map[string]any{
					"type": "content_block_start", "index": thinkingAt,
					"content_block": map[string]any{
						"type": "thinking", "thinking": "", "signature": ""}})
			}
			_ = anthropicSSE(w, flusher, "content_block_delta", map[string]any{
				"type": "content_block_delta", "index": thinkingAt,
				"delta": map[string]any{"type": "thinking_delta", "thinking": d.ReasoningContent}})
			wrote = true
		}

		if d.Content != "" {
			// A text block may not follow an open thinking block in some clients, so
			// close thinking before the answer starts.
			if thinkingAt >= 0 {
				closeBlock(thinkingAt)
				thinkingAt = -1
			}
			if textAt < 0 {
				textAt = nextIndex
				nextIndex++
				_ = anthropicSSE(w, flusher, "content_block_start", map[string]any{
					"type": "content_block_start", "index": textAt,
					"content_block": map[string]any{"type": "text", "text": ""}})
			}
			_ = anthropicSSE(w, flusher, "content_block_delta", map[string]any{
				"type": "content_block_delta", "index": textAt,
				"delta": map[string]any{"type": "text_delta", "text": d.Content}})
			wrote = true
		}

		for _, tc := range d.ToolCalls {
			if thinkingAt >= 0 {
				closeBlock(thinkingAt)
				thinkingAt = -1
			}
			if textAt >= 0 {
				closeBlock(textAt)
				textAt = -1
			}
			sawTools = true
			idx, seen := toolAt[tc.Index]
			if !seen {
				idx = nextIndex
				nextIndex++
				toolAt[tc.Index] = idx
				input := json.RawMessage("{}")
				id := tc.ID
				if id == "" {
					id = "toolu_" + strings.TrimPrefix(newMessageID(), "msg_")
				}
				_ = anthropicSSE(w, flusher, "content_block_start", map[string]any{
					"type": "content_block_start", "index": idx,
					"content_block": map[string]any{
						"type": "tool_use", "id": id, "name": tc.Function.Name,
						"input": input,
					}})
			}
			if tc.Function.Arguments != "" {
				_ = anthropicSSE(w, flusher, "content_block_delta", map[string]any{
					"type": "content_block_delta", "index": idx,
					"delta": map[string]any{
						"type": "input_json_delta", "partial_json": tc.Function.Arguments}})
			}
			wrote = true
		}
	}

	if thinkingAt >= 0 {
		closeBlock(thinkingAt)
	}
	if textAt >= 0 {
		closeBlock(textAt)
	}
	for _, idx := range toolAt {
		closeBlock(idx)
	}
	if !wrote {
		// An entirely empty upstream stream would leave a client waiting forever.
		// One empty text block is the honest minimum.
		_ = anthropicSSE(w, flusher, "content_block_start", map[string]any{
			"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "text", "text": ""}})
		closeBlock(0)
	}

	usage := map[string]any{"output_tokens": 0}
	if captured != nil {
		usage["output_tokens"] = captured.CompletionTokens
		usage["input_tokens"] = captured.PromptTokens
	}
	_ = anthropicSSE(w, flusher, "message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": anthropicStopReason(finish, sawTools), "stop_sequence": nil},
		"usage": usage,
	})
	_ = anthropicSSE(w, flusher, "message_stop", map[string]any{"type": "message_stop"})

	cost := record(name, model, captured, now)
	log.Printf("anthropic/messages model=%s provider=%s usd=%.6f", model, name, cost)
	return nil
}

// handleCountTokens answers the Anthropic token-counting route. It measures rather
// than estimates: a one-token completion on the same provider returns the real
// prompt_tokens. That costs a single output token instead of being wrong.
func handleCountTokens(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		anthropicError(w, 400, "invalid_request_error", err.Error())
		return
	}
	var req anthropicRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		anthropicError(w, 400, "invalid_request_error", "bad json: "+err.Error())
		return
	}
	model, ok := resolveModel(req.Model)
	if !ok {
		anthropicError(w, 400, "invalid_request_error", "unknown model: "+req.Model)
		return
	}
	// Server tools never reach the upstream: they are executed by this router, and a
	// token count for the request does not include work the API would do.
	functions, _, splitErr := splitTools(req.Tools)
	if splitErr != nil {
		anthropicError(w, 400, "invalid_request_error", splitErr.Error())
		return
	}
	req.Tools = functions
	payload, err := toOpenAIChat(&req)
	if err != nil {
		anthropicError(w, 400, "invalid_request_error", err.Error())
		return
	}
	payload["max_tokens"] = 1

	now := nowFn()
	order := candidateOrder(now, led.Spent(now), model)
	var attempts []map[string]string
	for _, name := range order {
		p := provs[name]
		key, keyErr := providerKey(r.Context(), name, p)
		if keyErr != nil || breakerOpen(name) {
			continue
		}
		body := make(map[string]any, len(payload)+2)
		for k, v := range payload {
			body[k] = v
		}
		body["model"] = upstreamModel(name, model)
		body["stream"] = false
		buf, _ := json.Marshal(body)
		reqUp, _ := http.NewRequestWithContext(r.Context(), http.MethodPost,
			p.Base+"/chat/completions", bytes.NewReader(buf))
		reqUp.Header.Set("Content-Type", "application/json")
		reqUp.Header.Set("Authorization", "Bearer "+key)
		ua := p.UA
		if ua == "" {
			ua = DefaultUserAgent
		}
		reqUp.Header.Set("User-Agent", ua)
		resp, err := httpc.Do(reqUp)
		if err != nil {
			attempts = append(attempts, map[string]string{"provider": name, "error": err.Error()})
			continue
		}
		rawResp, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		resp.Body.Close()
		if resp.StatusCode >= 400 {
			attempts = append(attempts, map[string]string{
				"provider": name, "error": fmt.Sprintf("HTTP %d", resp.StatusCode)})
			continue
		}
		var env usageEnvelope
		if json.Unmarshal(rawResp, &env) != nil || env.Usage == nil {
			attempts = append(attempts, map[string]string{"provider": name, "error": "no usage in reply"})
			continue
		}
		record(name, model, env.Usage, now)
		w.Header().Set("X-Router-Provider", name)
		writeJSON(w, 200, map[string]any{"input_tokens": env.Usage.PromptTokens})
		return
	}
	anthropicError(w, 502, "api_error", "count_tokens failed: "+summariseAttempts(attempts))
}
