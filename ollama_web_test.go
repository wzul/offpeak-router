package main

// Ollama Cloud names web search and fetch /api/web_search and /api/web_fetch and
// returns its own body shape. A client built on the ollama SDK parses exactly
// that, so these tests guard the field names and the array-not-null detail as
// hard contracts, not cosmetics.

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strings"
	"testing"
)

func TestOllamaWebSearchLimit(t *testing.T) {
	cases := []struct{ in, want int }{
		{0, defaultSearchLimit}, // absent means default, not zero results
		{-3, defaultSearchLimit},
		{3, 3},
		{10, 10},
		{50, ollamaWebSearchMax}, // Ollama caps at 10; do not exceed the vendor
	}
	for _, c := range cases {
		if got := ollamaSearchLimit(c.in); got != c.want {
			t.Errorf("ollamaSearchLimit(%d) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestOllamaWebSearchResultsShape(t *testing.T) {
	out := ollamaSearchResults([]webHit{
		{Title: "T1", URL: "https://a.example/1", Description: "snippet one", Position: 1},
	})
	if len(out) != 1 {
		t.Fatalf("want 1 result, got %d", len(out))
	}
	for _, k := range []string{"title", "url", "content"} {
		if _, ok := out[0][k]; !ok {
			t.Errorf("missing %q, which the ollama SDK reads", k)
		}
	}
	if out[0]["content"] != "snippet one" {
		t.Errorf("snippet must land in content, got %v", out[0]["content"])
	}
	// Vendor-only fields would tempt a caller into depending on them.
	for _, k := range []string{"description", "position"} {
		if _, ok := out[0][k]; ok {
			t.Errorf("%q is not part of Ollama's shape and must not be sent", k)
		}
	}
	// An empty search must still be an array, never null.
	if got := ollamaSearchResults(nil); got == nil {
		t.Error("empty results must serialise as [], not null")
	}
}

func TestOllamaWebFetchBodyShape(t *testing.T) {
	body := ollamaFetchBody(fetchDoc{Title: "Page", Content: "text"})
	for _, k := range []string{"title", "content", "links"} {
		if _, ok := body[k]; !ok {
			t.Errorf("missing %q, which the ollama SDK reads", k)
		}
	}
	links, _ := body["links"].([]string)
	if links == nil {
		t.Fatal("links must be an empty array, not null: a client iterating null crashes")
	}
	if len(links) != 0 {
		t.Errorf("want no links, got %v", links)
	}
	// The raw bytes must carry [] rather than null.
	b, _ := json.Marshal(body)
	if !strings.Contains(string(b), `"links":[]`) {
		t.Errorf("serialised body lacks an empty links array: %s", b)
	}
}

func TestExtractLinksResolvesAndDedupes(t *testing.T) {
	page := `<a href="/a">a</a><a href="b.html">b</a>` +
		`<a href="https://other.example/x#frag">x</a>` +
		`<a href="/a">a again</a><a href="mailto:x@y.z">mail</a><a href="#top">top</a>`
	base, _ := url.Parse("https://site.example/dir/page")

	got := extractLinks(page, base)
	want := []string{
		"https://site.example/a",
		"https://site.example/dir/b.html",
		"https://other.example/x",
	}
	if len(got) != len(want) {
		t.Fatalf("links = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("link %d = %q, want %q", i, got[i], want[i])
		}
	}
	for _, l := range got {
		if strings.Contains(l, "#") {
			t.Errorf("fragment kept in %q", l)
		}
		if strings.HasPrefix(l, "mailto:") {
			t.Errorf("non-http scheme kept: %q", l)
		}
	}
}

// Assets are not links a reader can follow; handing back a font host or an
// analytics beacon makes a model cite junk.
func TestExtractLinksIgnoresNonAnchorHrefs(t *testing.T) {
	page := `<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=x">` +
		`<link rel="preload" href="https://www.googletagmanager.com/gtm.js">` +
		`<script src="https://cdn.example/x.js"></script>` +
		`<a href="https://real.example/page">real</a>`
	base, _ := url.Parse("https://site.example/")

	got := extractLinks(page, base)
	if len(got) != 1 || got[0] != "https://real.example/page" {
		t.Fatalf("links = %v, want only the anchor", got)
	}
}

func TestExtractLinksCapsOutput(t *testing.T) {
	var sb strings.Builder
	for i := 0; i < maxLinks+25; i++ {
		sb.WriteString(`<a href="/p`)
		sb.WriteString(strings.Repeat("x", 1))
		sb.WriteString(itoa(i))
		sb.WriteString(`">l</a>`)
	}
	base, _ := url.Parse("https://site.example/")
	if got := extractLinks(sb.String(), base); len(got) != maxLinks {
		t.Errorf("links = %d, want the %d cap", len(got), maxLinks)
	}
}

// The fetch guard must run on this route too: an Ollama-named path is still an
// authenticated fetch on a host with internal services.
func TestOllamaWebFetchRefusesInternal(t *testing.T) {
	_, srv := newUpstream(t)
	router := setup(t, srv)

	// Loopback is refused on every host; a docker-bridge address only exists on
	// the fleet host, so using one here would make the test host-dependent.
	resp, body := postJSON(t, router.URL+"/api/web_fetch",
		`{"url":"http://127.0.0.1:8788/status"}`)
	if resp.StatusCode != 400 {
		t.Fatalf("status = %d, want 400; body = %s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "non-public") {
		t.Errorf("refusal reason lost: %s", body)
	}
}

func TestOllamaWebRoutesValidateInput(t *testing.T) {
	_, srv := newUpstream(t)
	router := setup(t, srv)

	if resp, _ := postJSON(t, router.URL+"/api/web_search", `{}`); resp.StatusCode != 400 {
		t.Errorf("search without a query = %d, want 400", resp.StatusCode)
	}
	if resp, _ := postJSON(t, router.URL+"/api/web_fetch", `{}`); resp.StatusCode != 400 {
		t.Errorf("fetch without a url = %d, want 400", resp.StatusCode)
	}
	// Ollama's own routes are POST-only; a GET must not silently succeed.
	if resp, body := http.Get(router.URL + "/api/web_search?q=x"); resp.StatusCode != 405 {
		t.Errorf("GET /api/web_search = %d, want 405; body = %s", resp.StatusCode, body)
	}
}

func TestOllamaWebRoutesRequireAuth(t *testing.T) {
	_, srv := newUpstream(t)
	router := setup(t, srv)
	authToken = "secret"
	t.Cleanup(func() { authToken = "" })

	for _, path := range []string{"/api/web_search", "/api/web_fetch"} {
		if resp, _ := postJSON(t, router.URL+path, `{}`); resp.StatusCode != 401 {
			t.Errorf("%s without a token = %d, want 401", path, resp.StatusCode)
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// A network failure is not the caller's mistake: the host resolved and was allowed, and
// nothing answered. Reporting 400 there tells a client to stop retrying a request that would
// work on the next attempt, and hides a dead upstream behind a client error.
func TestOllamaWebFetchReportsNetworkFailureAsUpstream(t *testing.T) {
	_, srv := newUpstream(t)
	router := setup(t, srv)

	prev := fetchOneFn
	t.Cleanup(func() { fetchOneFn = prev })
	fetchOneFn = func(ctx context.Context, target string) fetchDoc {
		return fetchDoc{URL: target, Error: "dial tcp 93.184.216.34:443: i/o timeout"}
	}

	resp, body := postJSON(t, router.URL+"/api/web_fetch", `{"url":"https://example.com/"}`)
	if resp.StatusCode != 502 {
		t.Fatalf("status = %d, want 502 for an unreachable host; body = %s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "i/o timeout") {
		t.Errorf("the reason was dropped: %s", body)
	}

	// A refused target stays a client error: the same route, a different verdict.
	fetchOneFn = func(ctx context.Context, target string) fetchDoc {
		return fetchDoc{URL: target, Refused: true, Error: "refusing to fetch 127.0.0.1: non-public address"}
	}
	resp, _ = postJSON(t, router.URL+"/api/web_fetch", `{"url":"http://127.0.0.1:8788/status"}`)
	if resp.StatusCode != 400 {
		t.Errorf("status = %d, want 400 for a refused target", resp.StatusCode)
	}
}
