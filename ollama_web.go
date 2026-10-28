package main

// Web search and fetch in Ollama Cloud's dialect.
//
// Ollama's own API exposes them at /api/web_search and /api/web_fetch with a
// different shape from the router's /v1 routes, so a client written against the
// ollama SDK (ollama.web_search / client.webSearch) needs those exact names and
// that exact body. Route names alone are not compatibility: the vendor returns
// results[].{title,url,content} and {title,content,links}, while /v1 returns
// data.web[] plus data[]. Both are served from one implementation.

import (
	"encoding/json"
	"io"
	"net/http"
)

// ollamaWebSearchMax is Ollama's own ceiling on max_results; accepting more here
// would let a client depend on behaviour the vendor does not have.
const ollamaWebSearchMax = 10

func handleOllamaWebSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, 405, map[string]any{"error": "use POST {query, max_results}"})
		return
	}
	var body struct {
		Query      string `json:"query"`
		MaxResults int    `json:"max_results"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<16)).Decode(&body); err != nil {
		writeJSON(w, 400, map[string]any{"error": "invalid JSON body: " + err.Error()})
		return
	}
	if body.Query == "" {
		writeJSON(w, 400, map[string]any{"error": "query is required"})
		return
	}
	limit := ollamaSearchLimit(body.MaxResults)

	hits, backend, err := routedSearch(r.Context(), body.Query, limit)
	searchCalls.Add(1)
	if err != nil {
		writeJSON(w, 502, map[string]any{"error": "search failed: " + err.Error()})
		return
	}

	if backend == "" {
		backend = searchBackendName()
	}
	w.Header().Set("X-Router-Backend", backend)
	writeJSON(w, 200, map[string]any{"results": ollamaSearchResults(hits)})
}

func handleOllamaWebFetch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, 405, map[string]any{"error": "use POST {url}"})
		return
	}
	var body struct {
		URL string `json:"url"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<16)).Decode(&body); err != nil {
		writeJSON(w, 400, map[string]any{"error": "invalid JSON body: " + err.Error()})
		return
	}
	if body.URL == "" {
		writeJSON(w, 400, map[string]any{"error": "url is required"})
		return
	}

	doc, backend := routedFetchNamed(r.Context(), body.URL)
	w.Header().Set("X-Router-Backend", backend)
	fetchCalls.Add(1)
	// A refused target is the caller's mistake, not an upstream fault: report it
	// as 400 so it is not retried, and keep the guard's wording.
	if doc.Error != "" && doc.Refused {
		writeJSON(w, 400, map[string]any{"error": doc.Error})
		return
	}
	if doc.Error != "" && doc.Status == 0 {
		// Resolved but unreachable: an upstream problem, not a malformed request.
		writeJSON(w, 502, map[string]any{"error": doc.Error})
		return
	}
	if doc.Error != "" && doc.Status >= 400 {
		writeJSON(w, doc.Status, map[string]any{"error": doc.Error})
		return
	}

	writeJSON(w, 200, ollamaFetchBody(doc))
}

// ollamaSearchLimit applies Ollama's own bounds: default 5, hard ceiling 10.
func ollamaSearchLimit(n int) int {
	if n <= 0 {
		return defaultSearchLimit
	}
	if n > ollamaWebSearchMax {
		return ollamaWebSearchMax
	}
	return n
}

// ollamaSearchResults renames the fields to the ones a vendor-SDK client reads.
// Ollama calls the snippet "content" and has no position field; emitting both
// spellings would let a caller depend on something the vendor never sends.
func ollamaSearchResults(hits []webHit) []map[string]any {
	results := make([]map[string]any, 0, len(hits))
	for _, h := range hits {
		results = append(results, map[string]any{
			"title":   h.Title,
			"url":     h.URL,
			"content": h.Description,
		})
	}
	return results
}

// ollamaFetchBody always carries a links array: Ollama sends [] rather than null,
// and a client iterating over null crashes instead of showing an empty list.
func ollamaFetchBody(doc fetchDoc) map[string]any {
	links := doc.Links
	if links == nil {
		links = []string{}
	}
	return map[string]any{
		"title":   doc.Title,
		"content": doc.Content,
		"links":   links,
	}
}
