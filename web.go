package main

// The web routes. Search and fetch are forwarded to the provider that owns them - see
// webupstream.go - so nothing here contacts the target.
//
// This file used to hold a DuckDuckGo scraper, an HTML-to-text converter, a link extractor and
// an HTTP fetcher of its own. That was a second implementation of a vendor feature, run from
// this host's egress, and the cost only showed up later: an SSRF guard and a dialer to
// maintain, no browser to render anything, and - on a box that resolves IPv6 without routing
// it - a ten-second timeout on every IPv6-capable site. A gateway that reimplements what it
// fronts is not more capable, it is another thing that can be broken separately.
//
// One check stays here, on the target alone: a caller's internal hostname should not be handed
// to a third party either, and refusing it costs no network call.

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"
)

const (
	defaultSearchLimit = 5
	maxSearchLimit     = 20
	maxFetchURLs       = 5
	maxFetchBytes      = 4 << 20
)

// Route counters, so /status can answer "is this route being used at all" for paths that leave
// no other trace: a forwarded call is billed to the account, not to this process.
var (
	searchCalls atomic.Int64
	fetchCalls  atomic.Int64
)

// webHit is one search result in this router's own shape (/v1/search).
type webHit struct {
	Title       string `json:"title"`
	URL         string `json:"url"`
	Description string `json:"description"`
	Position    int    `json:"position,omitempty"`
}

// fetchDoc is one fetched page. Refused marks a target this router will not even forward;
// everything else that can go wrong is the upstream's answer.
type fetchDoc struct {
	URL     string   `json:"url"`
	Title   string   `json:"title"`
	Content string   `json:"content"`
	Status  int      `json:"status"`
	Links   []string `json:"links,omitempty"`
	Error   string   `json:"error,omitempty"`
	// Refused is not serialised: callers act on the status code, and a field naming our own
	// policy invites a client to depend on it.
	Refused bool `json:"-"`
}

// handleSearch serves GET /v1/search?q= and POST /v1/search {query,limit}.
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

	hits, backend, err := routedSearch(r.Context(), q, limit)
	if err != nil {
		writeJSON(w, 502, map[string]any{"success": false,
			"error": map[string]string{"message": err.Error(), "backend": backend}})
		return
	}
	searchCalls.Add(1)
	w.Header().Set("X-Router-Backend", backend)
	writeJSON(w, 200, map[string]any{"success": true, "query": q, "data": map[string]any{"web": hits}})
}

// handleFetch serves GET /v1/fetch?url= and POST /v1/fetch {url|urls}.
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
	failed := 0
	for _, t := range targets {
		doc := routedFetch(r.Context(), t)
		if doc.Error != "" {
			failed++
		}
		docs = append(docs, doc)
	}
	fetchCalls.Add(1)

	// A failed fetch is not a successful response carrying empty strings. Every HTTP client
	// reads the status first; a 200 here means the caller keeps an empty document and never
	// learns why.
	status := http.StatusOK
	success := true
	if failed > 0 {
		success = false
	}
	if failed == len(docs) {
		// Nothing was fetched: say so with a status a client cannot ignore. One dead URL among
		// several is a partial result and keeps 200, with success:false and the errors listed.
		status = http.StatusBadGateway
		for _, d := range docs {
			if d.Refused {
				// A target this router will never forward is a request error: repeating it is
				// pointless, and 502 would invite a retry.
				status = http.StatusBadRequest
				break
			}
		}
	}
	body := map[string]any{"success": success, "data": docs}
	if failed > 0 {
		body["failed"] = failed
	}
	writeJSON(w, status, body)
}

// refuseTarget rejects a target this router will not forward. It looks only at the URL and at
// an IP literal in the host: no DNS, no dialer. Resolving to decide would put this process back
// in the business of connecting, and the upstream is the one that has to fetch anyway.
func refuseTarget(u *url.URL) error {
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("unsupported scheme %q", u.Scheme)
	}
	host := u.Hostname()
	if host == "" {
		return fmt.Errorf("missing host")
	}
	if ip := net.ParseIP(host); ip != nil {
		if isInternalIP(ip) {
			return fmt.Errorf("refusing to fetch %s: non-public address", host)
		}
		return nil
	}
	lower := strings.ToLower(strings.TrimSuffix(host, "."))
	if lower == "localhost" || strings.HasSuffix(lower, ".localhost") ||
		strings.HasSuffix(lower, ".local") || strings.HasSuffix(lower, ".internal") {
		return fmt.Errorf("refusing to fetch %s: non-public address", host)
	}
	return nil
}

// isInternalIP reports whether an address literal is one a caller has no business asking a
// third party to fetch.
func isInternalIP(ip net.IP) bool {
	if ip == nil {
		return true
	}
	if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() || ip.IsUnspecified() || ip.IsMulticast() {
		return true
	}
	// Carrier-grade NAT, and the IPv4 link-local metadata range.
	if v4 := ip.To4(); v4 != nil {
		if v4[0] == 100 && v4[1] >= 64 && v4[1] <= 127 {
			return true
		}
		if v4[0] == 169 && v4[1] == 254 {
			return true
		}
	}
	return false
}
