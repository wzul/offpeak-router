package main

// Server tools.
//
// Anthropic's web search and web fetch are not functions the client runs: the tool
// definition travels in the request and the API is expected to execute it. A client
// like Claude Code asks the model a question, and if the model wants to search, it is
// this side of the wire that must actually search and hand the results back - inside
// the same request.
//
// So a server tool cannot be forwarded as a function tool. Doing that returns a
// tool_use block to a client that has nothing to execute, and the session stalls. This
// file turns each server tool into a function the model can call, runs the loop here,
// and then renders the blocks Anthropic clients expect: server_tool_use paired with
// web_search_tool_result / web_fetch_tool_result.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

type serverToolKind int

const (
	serverToolNone serverToolKind = iota
	serverToolSearch
	serverToolFetch
)

// Synthetic function names. They match the server tool names Anthropic uses, so the
// conversation history a client sends back reads the same way in both directions.
const (
	serverToolSearchName = "web_search"
	serverToolFetchName  = "web_fetch"
)

// searchForModel and fetchForModel are what a server tool calls. Variables so a test can
// answer them directly: the real implementations run behind the SSRF guard, which refuses
// loopback by design and must not be loosened for the sake of a test.
var (
	searchForModel = runSearch
	fetchForModel  = fetchOne
)

// defaultSearchResults is how many hits one search hands the model. Anthropic returns
// up to ten; five keeps a search-heavy turn from crowding out the actual work.
const defaultSearchResults = 5

// maxServerToolRounds bounds the loop. A model that keeps searching would otherwise
// hold the request open until the client gives up.
const maxServerToolRounds = 8

// serverToolSpec is one server tool as the client declared it.
type serverToolSpec struct {
	Kind           serverToolKind
	Type           string
	Name           string
	MaxUses        int
	AllowedDomains []string
	BlockedDomains []string
	Searches       int // searches already performed for this request
	Fetches        int
}

// kindForServerToolType classifies a declared tool type. Custom (function) tools have
// an empty type or "custom".
func kindForServerToolType(t string) serverToolKind {
	switch {
	case strings.HasPrefix(t, "web_search"):
		return serverToolSearch
	case strings.HasPrefix(t, "web_fetch"):
		return serverToolFetch
	}
	return serverToolNone
}

// splitTools separates the declared tools into client functions and server tools.
// A server tool type this router cannot execute is an error rather than a silent drop:
// the client believes the capability is available.
func splitTools(tools []anthropicTool) (functions []anthropicTool, servers []serverToolSpec, err error) {
	for _, t := range tools {
		kind := kindForServerToolType(t.Type)
		if kind == serverToolNone {
			if t.Type != "" && t.Type != "custom" {
				return nil, nil, fmt.Errorf(
					"unsupported server tool type %q; this router executes web_search and web_fetch only", t.Type)
			}
			functions = append(functions, t)
			continue
		}
		name := t.Name
		if name == "" {
			name = serverToolSearchName
			if kind == serverToolFetch {
				name = serverToolFetchName
			}
		}
		servers = append(servers, serverToolSpec{
			Kind: kind, Type: t.Type, Name: name, MaxUses: t.MaxUses,
			AllowedDomains: t.AllowedDomains, BlockedDomains: t.BlockedDomains,
		})
	}
	return functions, servers, nil
}

func (s serverToolSpec) syntheticName() string {
	if s.Kind == serverToolFetch {
		return serverToolFetchName
	}
	return serverToolSearchName
}

// syntheticTools renders the server tools as functions the upstream model can call,
// in the OpenAI shape the rest of the router already speaks.
func syntheticTools(servers []serverToolSpec) []any {
	seen := map[string]bool{}
	out := make([]any, 0, len(servers))
	for _, s := range servers {
		name := s.syntheticName()
		if seen[name] {
			continue
		}
		seen[name] = true
		var params map[string]any
		switch s.Kind {
		case serverToolSearch:
			params = map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "The search query.",
					},
				},
				"required": []string{"query"},
			}
		default:
			params = map[string]any{
				"type": "object",
				"properties": map[string]any{
					"url": map[string]any{
						"type":        "string",
						"description": "The absolute URL to fetch.",
					},
				},
				"required": []string{"url"},
			}
		}
		desc := "Search the public web for current information. Returns titles, URLs and snippets."
		if s.Kind == serverToolFetch {
			desc = "Fetch one URL and return its readable text."
		}
		out = append(out, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": name, "description": desc, "parameters": params,
			},
		})
	}
	return out
}

// serverResult is one search hit in the shape Anthropic documents.
//
// encrypted_content is ours, not Anthropic's: it carries the snippet so that a later
// turn can restore what the model actually read. The client sends those blocks back
// verbatim, and dropping the field would leave the conversation with URLs and no text.
type serverResult struct {
	Type             string `json:"type"`
	URL              string `json:"url"`
	Title            string `json:"title"`
	EncryptedContent string `json:"encrypted_content,omitempty"`
	PageAge          string `json:"page_age,omitempty"`
}

type encryptedPayload struct {
	URL     string `json:"url"`
	Title   string `json:"title"`
	Snippet string `json:"snippet"`
}

func sealSnippet(url, title, snippet string) string {
	buf, err := json.Marshal(encryptedPayload{URL: url, Title: title, Snippet: snippet})
	if err != nil {
		return ""
	}
	return base64.StdEncoding.EncodeToString(buf)
}

func openSnippet(sealed string) (encryptedPayload, bool) {
	var p encryptedPayload
	raw, err := base64.StdEncoding.DecodeString(sealed)
	if err != nil {
		return p, false
	}
	if json.Unmarshal(raw, &p) != nil {
		return p, false
	}
	return p, true
}

// serverToolCall is one executed (or refused) server tool invocation.
type serverToolCall struct {
	ID      string
	Name    string
	Kind    serverToolKind
	Input   map[string]any
	Results []serverResult
	Doc     *fetchDoc
	Error   string // Anthropic error code, e.g. max_uses_exceeded
}

// blocks renders the pair of content blocks Anthropic clients expect for one call.
func (c serverToolCall) blocks() []any {
	use := map[string]any{
		"type": "server_tool_use", "id": c.ID, "name": c.Name,
		"input": c.Input,
	}
	if c.Kind == serverToolFetch {
		body := map[string]any{
			"type":        "web_fetch_tool_result",
			"tool_use_id": c.ID,
		}
		if c.Error != "" {
			body["content"] = map[string]any{
				"type": "web_fetch_tool_result_error", "error_code": c.Error}
			return []any{use, body}
		}
		payload := map[string]any{
			"type": "web_fetch_result",
			"url":  c.Input["url"],
			"content": map[string]any{
				"type": "document",
				"source": map[string]any{
					"type": "text", "media_type": "text/plain", "data": c.Doc.Content,
				},
				"title": c.Doc.Title,
			},
			"retrieved_at": time.Now().UTC().Format(time.RFC3339),
		}
		body["content"] = payload
		return []any{use, body}
	}

	body := map[string]any{"type": "web_search_tool_result", "tool_use_id": c.ID}
	if c.Error != "" {
		// Documented shape: the error replaces the result list.
		body["content"] = map[string]any{
			"type": "web_search_tool_result_error", "error_code": c.Error}
		return []any{use, body}
	}
	results := make([]any, 0, len(c.Results))
	for _, r := range c.Results {
		results = append(results, r)
	}
	body["content"] = results
	return []any{use, body}
}

// toolMessage is the upstream-facing view of the same call: the model sees what the
// search returned so it can answer.
func (c serverToolCall) toolMessage() map[string]any {
	var text string
	switch {
	case c.Error != "":
		text = c.Name + " failed: " + c.Error
	case c.Kind == serverToolFetch:
		text = fmt.Sprintf("Fetched %s\nTitle: %s\n\n%s", c.Doc.URL, c.Doc.Title, clip(c.Doc.Content, 20000))
	default:
		var b strings.Builder
		for _, r := range c.Results {
			snippet := ""
			if p, ok := openSnippet(r.EncryptedContent); ok {
				snippet = p.Snippet
			}
			fmt.Fprintf(&b, "%s\n%s\n%s\n\n", r.Title, r.URL, snippet)
		}
		text = strings.TrimSpace(b.String())
		if text == "" {
			text = "No results."
		}
	}
	return map[string]any{"role": "tool", "tool_call_id": c.ID, "content": text}
}

func clip(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "\n[truncated]"
}

// executeServerTool runs one call against the router's own web layer, honouring the
// declared limits. It never returns an error: a failed search is reported to the model
// as a failed tool call, which is what a real provider does too.
func executeServerTool(ctx context.Context, servers []serverToolSpec, id, name string, input map[string]any) serverToolCall {
	call := serverToolCall{ID: id, Name: name, Input: input}

	// Match the call to the declaration that allowed it. A model can only call names we
	// declared, but a client may declare both kinds with different limits.
	var spec *serverToolSpec
	for i := range servers {
		if servers[i].syntheticName() == name {
			spec = &servers[i]
			break
		}
	}
	if spec == nil {
		call.Error = "unavailable"
		return call
	}
	call.Kind = spec.Kind

	limitReached := func() bool {
		return spec.MaxUses > 0 && spec.Searches+spec.Fetches >= spec.MaxUses
	}

	switch spec.Kind {
	case serverToolSearch:
		query, _ := input["query"].(string)
		if strings.TrimSpace(query) == "" {
			call.Error = "invalid_tool_input"
			return call
		}
		if limitReached() {
			call.Error = "max_uses_exceeded"
			return call
		}
		hits, _, err := searchForModel(ctx, query, defaultSearchResults)
		if err != nil {
			call.Error = "unavailable"
			return call
		}
		spec.Searches++
		searchCalls.Add(1)
		for _, h := range hits {
			if !domainAllowed(spec, h.URL) {
				continue
			}
			call.Results = append(call.Results, serverResult{
				Type: "web_search_result", URL: h.URL, Title: h.Title,
				EncryptedContent: sealSnippet(h.URL, h.Title, h.Description),
			})
		}
		// An empty result list is a legitimate answer, not an error: the model should
		// say what it could not find rather than retry the same query.
		return call
	default:
		target, _ := input["url"].(string)
		if strings.TrimSpace(target) == "" {
			call.Error = "invalid_tool_input"
			return call
		}
		if limitReached() {
			call.Error = "max_uses_exceeded"
			return call
		}
		doc := fetchForModel(ctx, target)
		if doc.Error != "" {
			call.Error = "unavailable"
			call.Doc = &doc
			return call
		}
		spec.Fetches++
		fetchCalls.Add(1)
		call.Doc = &doc
		return call
	}
}

// domainAllowed applies the declared domain filters. Docs are bare domains with an
// optional path; allowed_domains and blocked_domains are mutually exclusive, but a
// client that sends both is not worth failing over - the block list wins.
func domainAllowed(spec *serverToolSpec, rawURL string) bool {
	host := strings.ToLower(strings.TrimSpace(rawURL))
	for _, prefix := range []string{"https://", "http://"} {
		host = strings.TrimPrefix(host, prefix)
	}
	if i := strings.IndexAny(host, "/?#"); i >= 0 {
		host = host[:i]
	}
	if i := strings.Index(host, "@"); i >= 0 {
		host = host[i+1:]
	}
	if i := strings.Index(host, ":"); i >= 0 {
		host = host[:i]
	}
	matches := func(list []string) bool {
		for _, d := range list {
			d = strings.ToLower(strings.TrimSpace(d))
			if d == "" {
				continue
			}
			if host == d || strings.HasSuffix(host, "."+d) {
				return true
			}
		}
		return false
	}
	if len(spec.BlockedDomains) > 0 && matches(spec.BlockedDomains) {
		return false
	}
	if len(spec.AllowedDomains) > 0 {
		return matches(spec.AllowedDomains)
	}
	return true
}

// historyText flattens a block a client sent back that came from a server tool. The
// upstream model needs the content, and the client's own block types mean nothing to it.
func historyText(blk anthropicBlock) string {
	var raw struct {
		Content json.RawMessage `json:"content"`
	}
	raw.Content = blk.Content
	switch blk.Type {
	case "web_search_tool_result":
		var results []serverResult
		if json.Unmarshal(blk.Content, &results) != nil {
			return "Search results unavailable."
		}
		var b strings.Builder
		for _, r := range results {
			snippet := ""
			if p, ok := openSnippet(r.EncryptedContent); ok {
				snippet = p.Snippet
			}
			fmt.Fprintf(&b, "%s\n%s\n%s\n\n", r.Title, r.URL, snippet)
		}
		if b.Len() == 0 {
			return "No results."
		}
		return strings.TrimSpace(b.String())
	case "web_fetch_tool_result":
		var doc struct {
			Content struct {
				Source struct {
					Data string `json:"data"`
				} `json:"source"`
				Title string `json:"title"`
			} `json:"content"`
			URL string `json:"url"`
		}
		if json.Unmarshal(blk.Content, &doc) != nil {
			return "Fetched content unavailable."
		}
		return fmt.Sprintf("Fetched %s\nTitle: %s\n\n%s", doc.URL, doc.Content.Title, clip(doc.Content.Source.Data, 20000))
	}
	return ""
}

// serverToolNames lists the synthetic names, so translation code can tell a server
// call from a client function call.
func serverToolNames() map[string]serverToolKind {
	return map[string]serverToolKind{
		serverToolSearchName: serverToolSearch,
		serverToolFetchName:  serverToolFetch,
	}
}
