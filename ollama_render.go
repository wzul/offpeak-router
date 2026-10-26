package main

// Ollama chat dialect, response side: provider selection, failover, billing and
// the translation of an OpenAI answer into Ollama's shape (NDJSON when
// streaming). Keeping this separate from the request mapping makes it obvious
// that routing, breakers and billing are the same code the /v1 route uses.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

func serveOllama(w http.ResponseWriter, r *http.Request, payload map[string]any,
	stream bool, model, clientModel, kind string) {

	now := nowFn()
	order := candidateOrder(now, led.Spent(now), model)

	if shadow {
		would := ""
		if len(order) > 0 {
			would = order[0]
		}
		writeJSON(w, 200, map[string]any{"shadow": true, "order": order,
			"would_use": would, "model": model, "dialect": "ollama/" + kind})
		return
	}
	if len(order) == 0 {
		writeJSON(w, 503, map[string]any{"error": "no provider can serve " + model + " right now"})
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
		// Headers must be set before anything is written; Set after WriteHeader is
		// silently ignored.
		if anyFailover {
			w.Header().Set("X-Router-Failover", "true")
		}
		if err := renderOllama(w, r, name, p, key, payload, stream, now, model, clientModel, kind); err == nil {
			breakerOK(name)
			return
		} else {
			log.Printf("provider %s failed (ollama/%s): %v", name, kind, err)
			breakerFail(name)
			anyFailover = true
			attempts = append(attempts, map[string]string{"provider": name, "error": err.Error()})
		}
	}
	writeJSON(w, 502, map[string]any{"error": "all providers failed", "attempts": attempts})
}

// renderOllama performs one upstream call and writes the answer in Ollama shape.
// It must not touch w until the upstream has answered, or failover is impossible.
func renderOllama(w http.ResponseWriter, r *http.Request, name string, p *Provider, key string,
	payload map[string]any, stream bool, now time.Time, model, clientModel, kind string) error {

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
		return renderOllamaOnce(w, resp.Body, name, model, clientModel, kind, now)
	}
	return renderOllamaStream(w, resp.Body, name, model, clientModel, kind, now)
}

func renderOllamaOnce(w http.ResponseWriter, body io.Reader, name, model, clientModel, kind string,
	now time.Time) error {

	raw, err := io.ReadAll(body)
	if err != nil {
		return fmt.Errorf("read body: %w", err)
	}
	var env struct {
		Choices []struct {
			Message struct {
				Content          string           `json:"content"`
				ReasoningContent string           `json:"reasoning_content"`
				ToolCalls        []map[string]any `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *usageBlock `json:"usage"`
	}
	if err := json.Unmarshal(raw, &env); err != nil {
		return fmt.Errorf("decode upstream: %w", err)
	}
	cost := record(name, model, env.Usage, now)

	out := ollamaEnvelope(clientModel)
	if len(env.Choices) > 0 {
		c := env.Choices[0]
		if kind == "generate" {
			out["response"] = c.Message.Content
			if c.Message.ReasoningContent != "" {
				out["thinking"] = c.Message.ReasoningContent
			}
		} else {
			msg := map[string]any{"role": "assistant", "content": c.Message.Content}
			if c.Message.ReasoningContent != "" {
				// Ollama calls this "thinking"; an OpenAI upstream calls it
				// reasoning_content. Dropping it makes a thinking model look mute,
				// because with a small token budget the content can be empty.
				msg["thinking"] = c.Message.ReasoningContent
			}
			if len(c.Message.ToolCalls) > 0 {
				msg["tool_calls"] = c.Message.ToolCalls
			}
			out["message"] = msg
		}
		out["done_reason"] = finishToOllama(c.FinishReason)
	}
	out["done"] = true
	if env.Usage != nil {
		out["prompt_eval_count"] = env.Usage.PromptTokens
		out["eval_count"] = env.Usage.CompletionTokens
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Router-Usd", fmt.Sprintf("%.6f", cost))
	writeJSON(w, 200, out)
	return nil
}

func renderOllamaStream(w http.ResponseWriter, body io.Reader, name, model, clientModel, kind string,
	now time.Time) error {

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(200)
	flusher, _ := w.(http.Flusher)
	enc := json.NewEncoder(w)

	var captured *usageBlock
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
					Content          string           `json:"content"`
					ReasoningContent string           `json:"reasoning_content"`
					ToolCalls        []map[string]any `json:"tool_calls"`
				} `json:"delta"`
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
		delta := chunk.Choices[0].Delta
		if delta.Content == "" && delta.ReasoningContent == "" && len(delta.ToolCalls) == 0 {
			continue
		}
		line := ollamaEnvelope(clientModel)
		if kind == "generate" {
			line["response"] = delta.Content
			if delta.ReasoningContent != "" {
				line["thinking"] = delta.ReasoningContent
			}
		} else {
			msg := map[string]any{"role": "assistant", "content": delta.Content}
			if delta.ReasoningContent != "" {
				// Reasoning-only chunks must still reach the client: with a tight
				// token budget they are the entire reply, and dropping them makes a
				// working model look like it hung.
				msg["thinking"] = delta.ReasoningContent
			}
			if len(delta.ToolCalls) > 0 {
				msg["tool_calls"] = delta.ToolCalls
			}
			line["message"] = msg
		}
		line["done"] = false
		if err := enc.Encode(line); err != nil {
			break
		}
		if flusher != nil {
			flusher.Flush()
		}
	}

	cost := record(name, model, captured, now)
	final := ollamaEnvelope(clientModel)
	if kind == "generate" {
		final["response"] = ""
	} else {
		final["message"] = map[string]any{"role": "assistant", "content": ""}
	}
	final["done"] = true
	final["done_reason"] = "stop"
	final["total_duration"] = time.Since(now).Nanoseconds()
	if captured != nil {
		final["prompt_eval_count"] = captured.PromptTokens
		final["eval_count"] = captured.CompletionTokens
		final["total_duration"] = time.Since(now).Nanoseconds()
	}
	_ = enc.Encode(final)
	if flusher != nil {
		flusher.Flush()
	}
	w.Header().Set("X-Router-Usd", fmt.Sprintf("%.6f", cost))
	return nil
}

// ollamaEnvelope is the part of every Ollama reply that identifies the model.
// The name echoes what the client asked for, in Ollama's name:tag form.
func ollamaEnvelope(clientModel string) map[string]any {
	model := clientModel
	if model == "" {
		model = defaultServed()
	}
	if !strings.Contains(model, ":") {
		model += ":latest"
	}
	return map[string]any{
		"model":      model,
		"created_at": time.Now().UTC().Format(time.RFC3339Nano),
	}
}

func finishToOllama(reason string) string {
	switch reason {
	case "length":
		return "length"
	case "stop", "":
		return "stop"
	default:
		return reason
	}
}
