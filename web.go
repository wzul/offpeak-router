package main

// Web capabilities for the router: /v1/search and /v1/fetch.
//
// These exist so a caller can point its whole web layer at one endpoint and get
// the same bearer auth and the same ledger as the model routes. The response
// shapes follow the tool-provider contract the Hermes web tools already speak
// ({success, data}), so a client can swap backends without a translator.
//
// Backends are chosen from what is configured: Brave when BRAVE_SEARCH_API_KEY is
// present, otherwise DuckDuckGo's HTML endpoint, which needs no key.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	defaultSearchLimit = 5
	maxSearchLimit     = 20
	maxFetchURLs       = 5
	maxFetchBytes      = 4 << 20
	webUserAgent       = "Mozilla/5.0 (compatible; offpeak-router/1.2; +https://router.wanzul-hosting.com)"
	searchTimeout      = 20 * time.Second
	fetchTimeout       = 25 * time.Second
)

var (
	searchCalls atomic.Int64
	fetchCalls  atomic.Int64
)

type webHit struct {
	Title       string `json:"title"`
	URL         string `json:"url"`
	Description string `json:"description"`
	Position    int    `json:"position"`
}

type fetchDoc struct {
	URL     string   `json:"url"`
	Title   string   `json:"title"`
	Content string   `json:"content"`
	Status  int      `json:"status"`
	Links   []string `json:"links,omitempty"`
	Error   string   `json:"error,omitempty"`
}

// webClient never follows a redirect into private space: the router sits on a
// host with internal services, so an unfiltered fetch route is an SSRF pivot.
func newWebClient() *http.Client {
	return &http.Client{
		Timeout: fetchTimeout,
		// Validate at connection time, not just before the request: a hostname
		// that re-resolves between the check and the dial would otherwise slip
		// through (DNS rebinding). The dialed address is the one that was checked.
		Transport: &http.Transport{
			Proxy:                 nil, // an environment proxy would bypass the guard
			DialContext:           safeDialContext,
			TLSHandshakeTimeout:   10 * time.Second,
			ResponseHeaderTimeout: 20 * time.Second,
			MaxIdleConns:          4,
			IdleConnTimeout:       30 * time.Second,
		},
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 5 {
				return errors.New("too many redirects")
			}
			return guardURL(req.URL)
		},
	}
}

var webClient = newWebClient()

// localNets holds every subnet this machine can reach without leaving the box -
// docker bridges included. RFC1918 alone is not enough: docker hands out ranges
// like 172.80.0.0/16, which is public address space and passes IsPrivate().
var (
	localNetsMu sync.RWMutex
	localNets   []*net.IPNet
	localNetsAt time.Time
	localNetTTL = 5 * time.Minute
)

func refreshLocalNets() {
	localNetsMu.RLock()
	fresh := time.Since(localNetsAt) < localNetTTL && localNets != nil
	localNetsMu.RUnlock()
	if fresh {
		return
	}
	var nets []*net.IPNet
	ifaces, err := net.Interfaces()
	if err == nil {
		for _, iface := range ifaces {
			addrs, err := iface.Addrs()
			if err != nil {
				continue
			}
			for _, a := range addrs {
				if n, ok := a.(*net.IPNet); ok {
					nets = append(nets, &net.IPNet{IP: n.IP.Mask(n.Mask), Mask: n.Mask})
				}
			}
		}
	}
	localNetsMu.Lock()
	localNets, localNetsAt = nets, time.Now()
	localNetsMu.Unlock()
}

func inLocalNet(ip net.IP) bool {
	refreshLocalNets()
	localNetsMu.RLock()
	defer localNetsMu.RUnlock()
	for _, n := range localNets {
		if n.Contains(ip) {
			return true
		}
	}
	return false
}

// safeDialContext resolves once, refuses non-public results, then dials the
// checked IP itself so the name cannot resolve differently a moment later.
func safeDialContext(ctx context.Context, network, addr string) (net.Conn, error) {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return nil, err
	}
	ips, err := net.DefaultResolver.LookupIPAddr(ctx, host)
	if err != nil {
		return nil, fmt.Errorf("resolve %s: %w", host, err)
	}
	if len(ips) == 0 {
		return nil, fmt.Errorf("resolve %s: no addresses", host)
	}
	chosen := ips[0].IP
	for _, cand := range ips {
		if !isBlockedIP(cand.IP) {
			chosen = cand.IP
			break
		}
	}
	if isBlockedIP(chosen) {
		return nil, fmt.Errorf("refusing to fetch %s: resolves to a non-public address", host)
	}
	return webDialer.DialContext(ctx, network, net.JoinHostPort(chosen.String(), port))
}

var webDialer = &net.Dialer{Timeout: 10 * time.Second}

// guardURL rejects anything that is not plain http(s) on a public address.
func guardURL(u *url.URL) error {
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("unsupported scheme %q", u.Scheme)
	}
	host := u.Hostname()
	if host == "" {
		return errors.New("missing host")
	}
	if ip := net.ParseIP(host); ip != nil {
		if isBlockedIP(ip) {
			return fmt.Errorf("refusing to fetch %s: non-public address", host)
		}
		return nil
	}
	ips, err := net.LookupIP(host)
	if err != nil {
		return fmt.Errorf("resolve %s: %w", host, err)
	}
	if len(ips) == 0 {
		return fmt.Errorf("resolve %s: no addresses", host)
	}
	for _, ip := range ips {
		if isBlockedIP(ip) {
			return fmt.Errorf("refusing to fetch %s: resolves to a non-public address", host)
		}
	}
	return nil
}

func isBlockedIP(ip net.IP) bool {
	if ip == nil {
		return true
	}
	if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() || ip.IsMulticast() || ip.IsUnspecified() {
		return true
	}
	// Carrier-grade NAT and the IPv4 metadata service.
	if v4 := ip.To4(); v4 != nil {
		if v4[0] == 100 && v4[1] >= 64 && v4[1] <= 127 {
			return true
		}
		if v4[0] == 169 && v4[1] == 254 {
			return true
		}
	}
	// Anything on a subnet this host itself sits on - docker bridges such as
	// 172.80.0.1 are outside RFC1918 and would otherwise be fair game.
	return inLocalNet(ip)
}

func handleSearch(w http.ResponseWriter, r *http.Request) {
	q, limit := "", defaultSearchLimit
	switch r.Method {
	case http.MethodGet:
		q = strings.TrimSpace(r.URL.Query().Get("q"))
		if q == "" {
			q = strings.TrimSpace(r.URL.Query().Get("query"))
		}
		if n := r.URL.Query().Get("limit"); n != "" {
			_, _ = fmt.Sscanf(n, "%d", &limit)
		}
	case http.MethodPost:
		var body struct {
			Query string `json:"query"`
			Q     string `json:"q"`
			Limit int    `json:"limit"`
		}
		if err := json.NewDecoder(io.LimitReader(r.Body, 1<<16)).Decode(&body); err != nil {
			writeJSON(w, 400, map[string]any{"success": false,
				"error": map[string]string{"message": "invalid JSON body: " + err.Error()}})
			return
		}
		q = strings.TrimSpace(body.Query)
		if q == "" {
			q = strings.TrimSpace(body.Q)
		}
		if body.Limit > 0 {
			limit = body.Limit
		}
	default:
		writeJSON(w, 405, map[string]any{"success": false,
			"error": map[string]string{"message": "use GET ?q= or POST {query}"}})
		return
	}
	if q == "" {
		writeJSON(w, 400, map[string]any{"success": false,
			"error": map[string]string{"message": "query is required"}})
		return
	}
	if limit < 1 {
		limit = defaultSearchLimit
	}
	if limit > maxSearchLimit {
		limit = maxSearchLimit
	}

	hits, backend, err := runSearch(r.Context(), q, limit)
	if err != nil {
		writeJSON(w, 502, map[string]any{"success": false,
			"error": map[string]string{"message": err.Error(), "backend": backend}})
		return
	}
	searchCalls.Add(1)
	w.Header().Set("X-Router-Backend", backend)
	writeJSON(w, 200, map[string]any{"success": true, "query": q, "data": map[string]any{"web": hits}})
}

// runSearch tries the configured backends in order and reports which answered.
func runSearch(ctx context.Context, q string, limit int) ([]webHit, string, error) {
	if key := os.Getenv("BRAVE_SEARCH_API_KEY"); key != "" {
		hits, err := braveSearch(ctx, key, q, limit)
		if err == nil && len(hits) > 0 {
			return hits, "brave", nil
		}
	}
	hits, err := ddgSearch(ctx, q, limit)
	if err != nil {
		return nil, "duckduckgo", err
	}
	if len(hits) == 0 {
		return nil, "duckduckgo", errors.New("search returned no results")
	}
	return hits, "duckduckgo", nil
}

func braveSearch(ctx context.Context, key, q string, limit int) ([]webHit, error) {
	u := "https://api.search.brave.com/res/v1/web/search?q=" + url.QueryEscape(q) +
		fmt.Sprintf("&count=%d", limit)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-Subscription-Token", key)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", webUserAgent)

	resp, err := webClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("brave: status %d", resp.StatusCode)
	}
	var out struct {
		Web struct {
			Results []struct {
				Title       string `json:"title"`
				URL         string `json:"url"`
				Description string `json:"description"`
			} `json:"results"`
		} `json:"web"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&out); err != nil {
		return nil, err
	}
	hits := make([]webHit, 0, len(out.Web.Results))
	for i, r := range out.Web.Results {
		if i >= limit {
			break
		}
		hits = append(hits, webHit{
			Title:       html.UnescapeString(stripTags(r.Title)),
			URL:         r.URL,
			Description: html.UnescapeString(stripTags(r.Description)),
			Position:    i + 1,
		})
	}
	return hits, nil
}

var (
	// DuckDuckGo's HTML endpoint has kept these class names for years, but the
	// attribute order varies, so match the class anywhere in the tag.
	ddgResultRe  = regexp.MustCompile(`(?is)<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>`)
	ddgSnippetRe = regexp.MustCompile(`(?is)<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>`)
	tagRe        = regexp.MustCompile(`(?s)<[^>]*>`)
	spaceRe      = regexp.MustCompile(`\s+`)
)

func ddgSearch(ctx context.Context, q string, limit int) ([]webHit, error) {
	form := url.Values{"q": {q}, "kl": {"wt-wt"}}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"https://html.duckduckgo.com/html/", strings.NewReader(form.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("User-Agent", webUserAgent)
	req.Header.Set("Accept", "text/html")

	resp, err := webClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("duckduckgo: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("duckduckgo: status %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if err != nil {
		return nil, err
	}
	return parseDDG(string(body), limit), nil
}

func parseDDG(page string, limit int) []webHit {
	links := ddgResultRe.FindAllStringSubmatch(page, -1)
	snips := ddgSnippetRe.FindAllStringSubmatch(page, -1)
	hits := make([]webHit, 0, len(links))
	seen := map[string]bool{}
	for i, m := range links {
		raw := html.UnescapeString(m[1])
		target := ddgTarget(raw)
		if target == "" || seen[target] {
			continue
		}
		desc := ""
		if i < len(snips) {
			desc = cleanText(snips[i][1])
		}
		seen[target] = true
		hits = append(hits, webHit{
			Title:       cleanText(m[2]),
			URL:         target,
			Description: desc,
			Position:    len(hits) + 1,
		})
		if len(hits) >= limit {
			break
		}
	}
	return hits
}

// ddgTarget unwraps DuckDuckGo's /l/?uddg=<encoded> redirect wrapper.
func ddgTarget(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	if strings.HasPrefix(raw, "//") {
		raw = "https:" + raw
	}
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	if target := u.Query().Get("uddg"); target != "" {
		if decoded, err := url.QueryUnescape(target); err == nil {
			return decoded
		}
		return target
	}
	if u.Scheme == "http" || u.Scheme == "https" {
		return u.String()
	}
	return ""
}

func handleFetch(w http.ResponseWriter, r *http.Request) {
	var targets []string
	switch r.Method {
	case http.MethodGet:
		if v := r.URL.Query().Get("url"); v != "" {
			targets = []string{v}
		}
	case http.MethodPost:
		var body struct {
			URL  string   `json:"url"`
			URLs []string `json:"urls"`
		}
		if err := json.NewDecoder(io.LimitReader(r.Body, 1<<18)).Decode(&body); err != nil {
			writeJSON(w, 400, map[string]any{"success": false,
				"error": map[string]string{"message": "invalid JSON body: " + err.Error()}})
			return
		}
		if body.URL != "" {
			targets = append(targets, body.URL)
		}
		targets = append(targets, body.URLs...)
	default:
		writeJSON(w, 405, map[string]any{"success": false,
			"error": map[string]string{"message": "use GET ?url= or POST {url|urls}"}})
		return
	}
	if len(targets) == 0 {
		writeJSON(w, 400, map[string]any{"success": false,
			"error": map[string]string{"message": "url or urls is required"}})
		return
	}
	if len(targets) > maxFetchURLs {
		targets = targets[:maxFetchURLs]
	}

	docs := make([]fetchDoc, 0, len(targets))
	for _, t := range targets {
		docs = append(docs, fetchOne(r.Context(), t))
	}
	fetchCalls.Add(1)
	writeJSON(w, 200, map[string]any{"success": true, "data": docs})
}

func fetchOne(ctx context.Context, target string) fetchDoc {
	doc := fetchDoc{URL: target}
	u, err := url.Parse(strings.TrimSpace(target))
	if err != nil {
		doc.Error = "invalid URL"
		return doc
	}
	if err := guardURL(u); err != nil {
		doc.Error = err.Error()
		return doc
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		doc.Error = err.Error()
		return doc
	}
	req.Header.Set("User-Agent", webUserAgent)
	req.Header.Set("Accept", "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5")

	resp, err := webClient.Do(req)
	if err != nil {
		doc.Error = err.Error()
		return doc
	}
	defer resp.Body.Close()
	doc.Status = resp.StatusCode

	ct := resp.Header.Get("Content-Type")
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxFetchBytes))
	if err != nil {
		doc.Error = "read body: " + err.Error()
		return doc
	}
	if strings.Contains(ct, "html") || (ct == "" && strings.Contains(string(raw[:min(512, len(raw))]), "<")) {
		doc.Title, doc.Content = htmlToText(string(raw))
		doc.Links = extractLinks(string(raw), u)
	} else {
		doc.Content = strings.TrimSpace(string(raw))
	}
	if resp.StatusCode >= 400 {
		doc.Error = fmt.Sprintf("upstream status %d", resp.StatusCode)
	}
	return doc
}

// htmlToText pulls the title and readable text out of a page without pulling in
// an HTML parser: drop non-content elements, turn block boundaries into breaks,
// strip the remaining tags, unescape entities, then collapse whitespace.
func htmlToText(page string) (title, text string) {
	if m := regexp.MustCompile(`(?is)<title[^>]*>(.*?)</title>`).FindStringSubmatch(page); len(m) == 2 {
		title = cleanText(m[1])
	}
	s := page
	for _, re := range []*regexp.Regexp{
		regexp.MustCompile(`(?is)<!--.*?-->`),
		regexp.MustCompile(`(?is)<script[^>]*>.*?</script>`),
		regexp.MustCompile(`(?is)<style[^>]*>.*?</style>`),
		regexp.MustCompile(`(?is)<noscript[^>]*>.*?</noscript>`),
		regexp.MustCompile(`(?is)<svg[^>]*>.*?</svg>`),
		regexp.MustCompile(`(?is)<head[^>]*>.*?</head>`),
	} {
		s = re.ReplaceAllString(s, " ")
	}
	// Block-level tags become line breaks so paragraphs stay distinguishable.
	s = regexp.MustCompile(`(?i)</?(p|div|section|article|header|footer|li|tr|h[1-6]|br|pre|blockquote)\b[^>]*>`).
		ReplaceAllString(s, "\n")
	s = tagRe.ReplaceAllString(s, " ")
	s = html.UnescapeString(s)
	lines := strings.Split(s, "\n")
	out := make([]string, 0, len(lines))
	for _, ln := range lines {
		if ln = strings.Join(strings.Fields(ln), " "); ln != "" {
			out = append(out, ln)
		}
	}
	return title, strings.Join(out, "\n")
}

// maxLinks caps the links a page can report: a large site's nav would otherwise
// bury the content a caller asked for.
const maxLinks = 100

// Anchors only: a bare href attribute also matches <link rel="stylesheet">, so a
// page's font and analytics hosts would arrive as "links" a model may then cite.
// The capture must include a fragment, or the closing quote stops matching and
// the whole anchor is skipped.
var hrefRe = regexp.MustCompile(`(?is)<a\b[^>]*?href\s*=\s*["']([^"']+)["']`)

// extractLinks pulls the page's anchors out as absolute http(s) URLs, in document
// order, without duplicates. Relative hrefs resolve against the page they came
// from, which is how a reader would follow them.
func extractLinks(page string, base *url.URL) []string {
	seen := map[string]bool{}
	var out []string
	for _, m := range hrefRe.FindAllStringSubmatch(page, -1) {
		raw := strings.TrimSpace(m[1])
		// A bare fragment is a link to the page you are already on; it is never
		// a reference worth handing back.
		if raw == "" || strings.HasPrefix(raw, "#") {
			continue
		}
		ref, err := url.Parse(raw)
		if err != nil {
			continue
		}
		abs := base.ResolveReference(ref)
		if abs.Scheme != "http" && abs.Scheme != "https" {
			continue
		}
		abs.Fragment = ""
		key := abs.String()
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, key)
		if len(out) >= maxLinks {
			break
		}
	}
	return out
}

func stripTags(s string) string { return strings.TrimSpace(tagRe.ReplaceAllString(s, "")) }

// searchBackendName reports which backend a search would use right now.
func searchBackendName() string {
	if os.Getenv("BRAVE_SEARCH_API_KEY") != "" {
		return "brave"
	}
	return "duckduckgo"
}

func cleanText(s string) string {
	return strings.TrimSpace(spaceRe.ReplaceAllString(html.UnescapeString(tagRe.ReplaceAllString(s, " ")), " "))
}
