package main

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

// A real DuckDuckGo HTML fragment: results wrapped in a /l/?uddg= redirect, an
// anchor snippet, and entities that must be decoded. If DDG changes its markup
// this fixture is the first thing to refresh.
const ddgFixture = `<html><body>
<div class="result results_links results_links_deep web-result">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev%2Fdoc%2F&amp;rut=abc">The Go Programming Language &amp; Docs</a>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev%2Fdoc%2F">Official documentation for <b>Go</b>.</a>
</div>
<div class="result results_links results_links_deep web-result">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fpkg.go.dev%2Fnet%2Fhttp">net/http package</a>
  <a class="result__snippet" href="#">HTTP client and server implementations.</a>
</div>
<div class="result results_links results_links_deep web-result">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev%2Fdoc%2F&amp;rut=dup">Duplicate that must be dropped</a>
</div>
</body></html>`

func TestParseDDGUnwrapsRedirectsAndDedupes(t *testing.T) {
	hits := parseDDG(ddgFixture, 10)
	if len(hits) != 2 {
		t.Fatalf("want 2 unique results, got %d: %+v", len(hits), hits)
	}
	if hits[0].URL != "https://go.dev/doc/" {
		t.Errorf("first URL = %q, want unwrapped https://go.dev/doc/", hits[0].URL)
	}
	if !strings.Contains(hits[0].Title, "Go Programming Language") {
		t.Errorf("title not decoded/stripped: %q", hits[0].Title)
	}
	if strings.Contains(hits[0].Title, "<b>") || strings.Contains(hits[0].Description, "<b>") {
		t.Errorf("tags leaked into text: %+v", hits[0])
	}
	if hits[1].URL != "https://pkg.go.dev/net/http" {
		t.Errorf("second URL = %q", hits[1].URL)
	}
	if hits[0].Position != 1 || hits[1].Position != 2 {
		t.Errorf("positions wrong: %d %d", hits[0].Position, hits[1].Position)
	}
}

func TestParseDDGRespectsLimit(t *testing.T) {
	if got := len(parseDDG(ddgFixture, 1)); got != 1 {
		t.Errorf("limit 1 returned %d results", got)
	}
}

func TestDDGTarget(t *testing.T) {
	cases := map[string]string{
		"//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fb%3D1": "https://example.com/a?b=1",
		"https://example.com/direct":                                     "https://example.com/direct",
		"javascript:alert(1)":                                            "",
		"":                                                               "",
	}
	for in, want := range cases {
		if got := ddgTarget(in); got != want {
			t.Errorf("ddgTarget(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestHTMLToTextDropsNoiseKeepsParagraphs(t *testing.T) {
	page := `<html><head><title>  Hello &amp; World </title>
	<style>.x{color:red}</style></head>
	<body><script>var leak = "secret";</script>
	<h1>Heading</h1><p>First   para with <a href="#">link</a>.</p>
	<div>Second line</div><noscript>nope</noscript></body></html>`
	title, text := htmlToText(page)
	if title != "Hello & World" {
		t.Errorf("title = %q", title)
	}
	for _, bad := range []string{"secret", "color:red", "nope", "<", ">"} {
		if strings.Contains(text, bad) {
			t.Errorf("text still contains %q: %q", bad, text)
		}
	}
	if !strings.Contains(text, "Heading") || !strings.Contains(text, "First para with link") {
		t.Errorf("content lost: %q", text)
	}
	if !strings.Contains(text, "\n") {
		t.Errorf("block boundaries not preserved: %q", text)
	}
}

func TestGuardURLBlocksInternalAddresses(t *testing.T) {
	blocked := []string{
		"http://127.0.0.1/", "http://localhost/", "http://10.0.0.5/",
		"http://192.168.1.1/", "http://172.16.0.1/", "http://169.254.169.254/latest/meta-data/",
		"http://[::1]/", "http://100.64.0.1/",
	}
	for _, raw := range blocked {
		u, err := url.Parse(raw)
		if err != nil {
			t.Fatalf("bad fixture %q: %v", raw, err)
		}
		if err := guardURL(u); err == nil {
			t.Errorf("%s should be refused", raw)
		}
	}
	for _, raw := range []string{"file:///etc/passwd", "gopher://example.com/", "ftp://example.com/"} {
		u, _ := url.Parse(raw)
		if err := guardURL(u); err == nil {
			t.Errorf("scheme of %s should be refused", raw)
		}
	}
}

func TestIsBlockedIP(t *testing.T) {
	for _, s := range []string{"127.0.0.1", "10.1.2.3", "172.20.0.1", "192.168.0.1",
		"169.254.169.254", "::1", "fd00::1", "100.100.100.100", "0.0.0.0"} {
		if !isBlockedIP(net.ParseIP(s)) {
			t.Errorf("%s should be blocked", s)
		}
	}
	for _, s := range []string{"1.1.1.1", "8.8.8.8", "2606:4700:4700::1111"} {
		if isBlockedIP(net.ParseIP(s)) {
			t.Errorf("%s should be allowed", s)
		}
	}
}

// handleFetch must never reach the network for an internal target, even though
// the endpoint is authenticated: an operator typo should not become an SSRF.
func TestHandleFetchRefusesInternalWithoutDialing(t *testing.T) {
	prev := webClient
	dialled := false
	webClient = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		dialled = true
		return nil, nil
	})}
	defer func() { webClient = prev }()

	rec := httptest.NewRecorder()
	body := strings.NewReader(`{"url":"http://169.254.169.254/latest/meta-data/"}`)
	handleFetch(rec, httptest.NewRequest(http.MethodPost, "/v1/fetch", body))

	if dialled {
		t.Fatal("request was dialled: SSRF guard bypassed")
	}
	// A refused target is a request error, and the response must say so: this used to be a 200
	// with an empty document, which every HTTP client reads as success.
	if rec.Code != 400 {
		t.Fatalf("status = %d, want 400 for a refused target", rec.Code)
	}
	var out struct {
		Success *bool      `json:"success"`
		Data    []fetchDoc `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Success == nil || *out.Success {
		t.Errorf("success = %v on a refused target", out.Success)
	}
	if len(out.Data) != 1 || out.Data[0].Error == "" {
		t.Fatalf("expected a per-URL error, got %+v", out.Data)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestHandleSearchValidatesInput(t *testing.T) {
	rec := httptest.NewRecorder()
	handleSearch(rec, httptest.NewRequest(http.MethodPost, "/v1/search", strings.NewReader(`{}`)))
	if rec.Code != 400 {
		t.Errorf("empty query should be 400, got %d", rec.Code)
	}
	rec = httptest.NewRecorder()
	handleSearch(rec, httptest.NewRequest(http.MethodPost, "/v1/search", strings.NewReader(`not json`)))
	if rec.Code != 400 {
		t.Errorf("bad JSON should be 400, got %d", rec.Code)
	}
	rec = httptest.NewRecorder()
	handleSearch(rec, httptest.NewRequest(http.MethodPut, "/v1/search", nil))
	if rec.Code != 405 {
		t.Errorf("PUT should be 405, got %d", rec.Code)
	}
}

// The routes must be behind the bearer gate like every other /v1 path.
func TestWebRoutesRequireAuth(t *testing.T) {
	prev := authToken
	authToken = "secret-token"
	defer func() { authToken = prev }()

	for _, path := range []string{"/v1/search?q=x", "/v1/fetch?url=https://example.com"} {
		rec := httptest.NewRecorder()
		newMux().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
		if rec.Code != 401 {
			t.Errorf("%s without a token = %d, want 401", path, rec.Code)
		}
	}
}

func TestFetchOneExtractsTextFromServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write([]byte(`<html><head><title>Doc</title><script>bad()</script></head>
			<body><p>Body text here.</p></body></html>`))
	}))
	defer srv.Close()

	// httptest listens on loopback, which the guard blocks by design, so bypass the
	// guard here and test only the extraction path.
	doc := fetchDirect(t, srv.URL)
	if doc.Status != 200 {
		t.Fatalf("status = %d (%s)", doc.Status, doc.Error)
	}
	if doc.Title != "Doc" || !strings.Contains(doc.Content, "Body text here.") {
		t.Fatalf("unexpected doc: %+v", doc)
	}
	if strings.Contains(doc.Content, "bad()") {
		t.Error("script content leaked")
	}
}

// fetchDirect runs the body-reading half of fetchOne against a test server.
func fetchDirect(t *testing.T, target string) fetchDoc {
	t.Helper()
	resp, err := http.Get(target) //nolint:gosec // test-only, target is a local httptest server
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	doc := fetchDoc{URL: target, Status: resp.StatusCode}
	if ct := resp.Header.Get("Content-Type"); strings.Contains(ct, "html") {
		raw, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		doc.Title, doc.Content = htmlToText(string(raw))
	}
	return doc
}

// Docker bridges are outside RFC1918 (172.80.0.0/16 is public space by the
// standard) yet reachable from the host, so the guard must also refuse anything
// on a subnet this machine sits on.
func TestIsBlockedIPRefusesLocalSubnet(t *testing.T) {
	_, n, _ := net.ParseCIDR("172.80.0.0/16")
	localNetsMu.Lock()
	prevNets, prevAt := localNets, localNetsAt
	localNets, localNetsAt = []*net.IPNet{n}, time.Now()
	localNetsMu.Unlock()
	defer func() {
		localNetsMu.Lock()
		localNets, localNetsAt = prevNets, prevAt
		localNetsMu.Unlock()
	}()

	if !isBlockedIP(net.ParseIP("172.80.0.1")) {
		t.Error("172.80.0.1 is on the host's own docker bridge and must be blocked")
	}
	if !isBlockedIP(net.ParseIP("172.80.1.14")) {
		t.Error("172.80.1.14 (a container) must be blocked")
	}
	if isBlockedIP(net.ParseIP("1.1.1.1")) {
		t.Error("1.1.1.1 is public and must be allowed")
	}
	if err := guardURL(&url.URL{Scheme: "http", Host: "172.80.0.1:8788", Path: "/status"}); err == nil {
		t.Error("guardURL let a local-bridge address through")
	}
}
