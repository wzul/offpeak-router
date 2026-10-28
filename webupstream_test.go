package main

// The router is a gateway for web search and fetch: the vendor owns those endpoints, and the
// local implementation is only the answer when no account can serve the call. These tests hold
// the order (pinned account first), the shapes every dialect already promises, and the fallback.

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// webRig stands up a fake ollama.com that records which account and path was used.
type webRig struct {
	srv   *httptest.Server
	mu    sync.Mutex
	hits  []string
	fail  bool
	empty bool
}

func (r *webRig) server(t *testing.T) {
	r.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		key := strings.TrimPrefix(req.Header.Get("Authorization"), "Bearer ")
		r.mu.Lock()
		r.hits = append(r.hits, key+" "+req.URL.Path)
		fail, empty := r.fail, r.empty
		r.mu.Unlock()

		if fail {
			w.WriteHeader(402)
			w.Write([]byte(`{"error":"payment required"}`))
			return
		}
		switch req.URL.Path {
		case "/api/web_search":
			if empty {
				w.Write([]byte(`{"results":[]}`))
				return
			}
			json.NewEncoder(w).Encode(map[string]any{"results": []any{map[string]any{
				"title": "Hidup", "url": "https://example.com/a", "content": "ringkasan dari ollama",
			}}})
		case "/api/web_fetch":
			if empty {
				w.Write([]byte(`{"title":"","content":""}`))
				return
			}
			json.NewEncoder(w).Encode(map[string]any{
				"title": "Hidup", "content": "isi halaman dari ollama", "links": []string{"https://example.com/b"},
			})
		default:
			w.WriteHeader(404)
		}
	}))
	t.Cleanup(r.srv.Close)
}

func (r *webRig) recorded() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.hits...)
}

func webRigProviders(t *testing.T, base string) {
	t.Helper()
	prev := provs
	t.Cleanup(func() { provs = prev })
	provs = map[string]*Provider{
		"ollama-lenjan": {Base: base + "/v1", KeyEnv: "SPARE_KEY", First: true, Web: true,
			Models: map[string]ModelRoute{"m": {ID: "m"}}},
		"ollama": {Base: base + "/v1", KeyEnv: "MAIN_KEY", Web: true,
			Models: map[string]ModelRoute{"m": {ID: "m"}}},
	}
	t.Setenv("SPARE_KEY", "spare-key")
	t.Setenv("MAIN_KEY", "main-key")
	breakers = map[string]*breakerEntry{}
}

func TestWebSearchGoesToThePinnedAccountFirst(t *testing.T) {
	rig := &webRig{}
	rig.server(t)
	webRigProviders(t, rig.srv.URL)

	hits, backend, err := routedSearch(context.Background(), "golang", 3)
	if err != nil {
		t.Fatalf("routedSearch: %v", err)
	}
	if backend != "ollama:ollama-lenjan" {
		t.Errorf("backend = %q, want the pinned account", backend)
	}
	if len(hits) != 1 || hits[0].Description != "ringkasan dari ollama" {
		t.Errorf("hits = %+v", hits)
	}
	if got := rig.recorded(); len(got) != 1 || !strings.HasPrefix(got[0], "spare-key") {
		t.Errorf("upstream calls = %v, want exactly one from the spare account", got)
	}
}

func TestWebFetchUsesTheUpstreamDocument(t *testing.T) {
	rig := &webRig{}
	rig.server(t)
	webRigProviders(t, rig.srv.URL)

	doc, backend := routedFetchNamed(context.Background(), "https://example.com/a")
	if backend != "ollama:ollama-lenjan" {
		t.Errorf("backend = %q", backend)
	}
	if doc.Error != "" {
		t.Fatalf("fetch error: %s", doc.Error)
	}
	if doc.Content != "isi halaman dari ollama" || doc.Title != "Hidup" {
		t.Errorf("doc = %+v", doc)
	}
	if len(doc.Links) != 1 || doc.Links[0] != "https://example.com/b" {
		t.Errorf("links = %v", doc.Links)
	}
}

// A held account must not be asked, for web either: it is the same credential.
func TestHeldAccountIsSkippedForWeb(t *testing.T) {
	rig := &webRig{}
	rig.server(t)
	webRigProviders(t, rig.srv.URL)

	breakerFailWith("ollama-lenjan", newUpstreamError(402, "payment required"))

	_, backend, err := routedSearch(context.Background(), "golang", 3)
	if err != nil {
		t.Fatalf("routedSearch: %v", err)
	}
	if backend != "ollama:ollama" {
		t.Errorf("backend = %q, want the next account", backend)
	}
	for _, h := range rig.recorded() {
		if strings.HasPrefix(h, "spare-key") {
			t.Errorf("the held account was called: %v", rig.recorded())
		}
	}
}

// An empty document is not an answer: fall back rather than hand the caller a blank page.
func TestBadURLDoesNotBlameTheAccount(t *testing.T) {
	prev := provs
	t.Cleanup(func() { provs = prev })

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(422)
		w.Write([]byte(`{"error":"could not resolve that host"}`))
	}))
	t.Cleanup(srv.Close)
	provs = map[string]*Provider{
		"ollama-lenjan": {Base: srv.URL + "/v1", KeyEnv: "SPARE_KEY", First: true, Web: true,
			Models: map[string]ModelRoute{"m": {ID: "m"}}},
	}
	t.Setenv("SPARE_KEY", "spare-key")
	breakers = map[string]*breakerEntry{}

	routedFetchNamed(context.Background(), "https://tiada.example/")

	brMu.Lock()
	entry := breakers["ollama-lenjan"]
	brMu.Unlock()
	if entry != nil && entry.fails > 0 {
		t.Errorf("a 422 for a bad URL counted %d failure(s) against the account", entry.fails)
	}

	// The same route with a billing answer must still park the account.
	srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(402)
		w.Write([]byte(`{"error":"payment required"}`))
	}))
	t.Cleanup(srv2.Close)
	provs["ollama-lenjan"] = &Provider{Base: srv2.URL + "/v1", KeyEnv: "SPARE_KEY", First: true,
		Web: true, Models: map[string]ModelRoute{"m": {ID: "m"}}}
	routedFetchNamed(context.Background(), "https://example.com/")
	if !breakerOpen("ollama-lenjan") {
		t.Error("a 402 on the web route did not hold the account")
	}
}

// /status must say which path serves web calls, so an operator can tell forwarding from the
// local fallback without reading the code.
func TestStatusNamesTheWebBackends(t *testing.T) {
	rig := &webRig{}
	rig.server(t)
	webRigProviders(t, rig.srv.URL)

	router := httptest.NewServer(newMux())
	t.Cleanup(router.Close)

	resp, err := http.Get(router.URL + "/status")
	if err != nil {
		t.Fatalf("get status: %v", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read status: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d: %s", resp.StatusCode, raw)
	}
	var out struct {
		Web struct {
			Backends string `json:"upstream_backends"`
		} `json:"web"`
	}
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !strings.Contains(out.Web.Backends, "ollama-lenjan") {
		t.Errorf("upstream_backends = %q, want the web-capable accounts named", out.Web.Backends)
	}
}

// A search response must stay a search response. The vendor returns whole-page text in its
// search results, and passing that through verbatim made a five-result query answer with a
// megabyte of page bodies.
func TestSearchDescriptionsAreSnippets(t *testing.T) {
	long := strings.Repeat("perkataan panjang ", 4000) // ~68k characters
	rig := &webRig{}
	rig.server(t)
	webRigProviders(t, rig.srv.URL)

	// Re-point the fake upstream at an oversized result.
	big := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"results": []any{map[string]any{
			"title": "Besar", "url": "https://example.com/besar", "content": long,
		}}})
	}))
	t.Cleanup(big.Close)
	provs["ollama-lenjan"] = &Provider{Base: big.URL + "/v1", KeyEnv: "SPARE_KEY", First: true,
		Web: true, Models: map[string]ModelRoute{"m": {ID: "m"}}}

	hits, _, err := routedSearch(context.Background(), "apa-apa", 5)
	if err != nil {
		t.Fatalf("routedSearch: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("hits = %d", len(hits))
	}
	if got := len(hits[0].Description); got > maxSnippetChars+8 {
		t.Errorf("description is %d characters, want a snippet near %d", got, maxSnippetChars)
	}
	if !strings.HasSuffix(hits[0].Description, "…") {
		t.Error("a truncated snippet should say it was cut")
	}
}
