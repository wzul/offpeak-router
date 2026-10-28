package main

// Web search and web fetch are ollama.com features. This router is a gateway: it should
// forward those two calls to the account that owns them, not perform them itself.
//
// The local implementation stays as the fallback, because a gateway that fails a request the
// account could not serve at that moment is worse than one that answers it - but the primary
// path is the vendor's own endpoint, tried per provider in the same pinned order as chat, so a
// spare account absorbs this traffic before the main one and an account's web access is billed
// to the same place its chat is. Doing the work locally also meant doing it from this host's
// egress: no IPv6, no browser, and an SSRF guard to maintain, all to reproduce an endpoint the
// vendor already runs.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
)

// webUpstreamProviders lists providers that expose the web endpoints, pinned first, then by
// name so the order is reproducible instead of following Go's map iteration.
func webUpstreamProviders() []string {
	var pinned, rest []string
	for name, p := range provs {
		if !p.Web {
			continue
		}
		if p.First {
			pinned = append(pinned, name)
		} else {
			rest = append(rest, name)
		}
	}
	sort.Strings(pinned)
	sort.Strings(rest)
	return append(pinned, rest...)
}

// webRequest posts to an ollama.com web endpoint on behalf of one provider.
func webRequest(ctx context.Context, name string, p *Provider, path string, body map[string]any) ([]byte, error) {
	key, err := providerKey(ctx, name, p)
	if err != nil {
		return nil, err
	}
	buf, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	base := strings.TrimSuffix(p.Base, "/v1")
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+path, bytes.NewReader(buf))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	// The web endpoints sit beside the chat API and are reached with the same client
	// identity; presenting a different one there would be a tell. There is no client
	// request on this path, so the header helper gets an empty one rather than a nil it
	// would dereference when a provider carries a session id.
	applyUpstreamHeaders(req, p, httptest.NewRequest(http.MethodPost, path, nil), body)

	resp, err := httpc.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxFetchBytes))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		return nil, newUpstreamError(resp.StatusCode, string(raw))
	}
	return raw, nil
}

// maxSnippetChars bounds a search result's description. Long enough to be a useful excerpt,
// short enough that a five-result search stays a search response.
const maxSnippetChars = 600

// snippet trims text on a word boundary and marks the cut, so a truncated excerpt cannot be
// mistaken for the end of the document.
func snippet(s string, limit int) string {
	s = strings.TrimSpace(strings.Join(strings.Fields(s), " "))
	if len(s) <= limit {
		return s
	}
	cut := s[:limit]
	if i := strings.LastIndexByte(cut, ' '); i > limit/2 {
		cut = cut[:i]
	}
	return cut + "…"
}

// accountBlame reports whether a web failure says something about the ACCOUNT. A malformed or
// unreachable URL is the caller's problem: counting it would let a handful of bad URLs open the
// breaker on healthy accounts, which is how a gateway parks itself for a caller's typo.
func accountBlame(err error) bool {
	var up *upstreamError
	if !errors.As(err, &up) {
		// Transport failure: no answer at all, which the account or the network could explain.
		return true
	}
	switch {
	case up.Status == http.StatusUnauthorized, up.Status == http.StatusPaymentRequired:
		return true
	case up.Status == http.StatusForbidden:
		hold, _ := holdsIndefinitely(err)
		return hold
	case up.Status == http.StatusTooManyRequests, up.Status >= 500:
		return true
	}
	return false
}

// routedSearch asks each web-capable provider in turn, then falls back to the local scraper.
// The returned string names whoever answered, so a caller can see which path ran.
func routedSearch(ctx context.Context, q string, limit int) ([]webHit, string, error) {
	var firstErr error
	for _, name := range webUpstreamProviders() {
		p := provs[name]
		if breakerOpen(name) {
			continue
		}
		raw, err := webRequest(ctx, name, p, "/api/web_search",
			map[string]any{"query": q, "max_results": limit})
		if err != nil {
			// The credential and the account are shared with chat, so an account-shaped
			// failure here is evidence about the account - but only then.
			if accountBlame(err) {
				breakerFailWith(name, err)
			}
			log.Printf("web search via %s failed: %v", name, err)
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		var out struct {
			Results []struct {
				Title   string `json:"title"`
				URL     string `json:"url"`
				Content string `json:"content"`
			} `json:"results"`
		}
		if err := json.Unmarshal(raw, &out); err != nil {
			if firstErr == nil {
				firstErr = fmt.Errorf("decode %s: %w", name, err)
			}
			continue
		}
		hits := make([]webHit, 0, len(out.Results))
		for _, r := range out.Results {
			// The vendor's search results carry whole-page text where this router's older
			// shape promised a description. /v1/search is the shape with a snippet, and a
			// caller asking for five results should not receive a megabyte of page bodies.
			hits = append(hits, webHit{Title: r.Title, URL: r.URL, Description: snippet(r.Content, maxSnippetChars)})
		}
		if len(hits) == 0 {
			if firstErr == nil {
				firstErr = fmt.Errorf("%s returned no results", name)
			}
			continue
		}
		return hits, "ollama:" + name, nil
	}

	hits, backend, err := localSearch(ctx, q, limit)
	if err != nil {
		if firstErr != nil {
			return nil, backend, fmt.Errorf("upstream: %v; local: %w", firstErr, err)
		}
		return nil, backend, err
	}
	return hits, backend, nil
}

// routedFetch forwards one page fetch to a web-capable provider, falling back to fetching it
// here. The document keeps the local shape so every caller and every dialect is unchanged.
func routedFetch(ctx context.Context, target string) fetchDoc {
	doc, _ := routedFetchNamed(ctx, target)
	return doc
}

// routedFetchNamed is the same call, reporting who answered.
func routedFetchNamed(ctx context.Context, target string) (fetchDoc, string) {
	var firstErr error
	for _, name := range webUpstreamProviders() {
		p := provs[name]
		if breakerOpen(name) {
			continue
		}
		raw, err := webRequest(ctx, name, p, "/api/web_fetch", map[string]any{"url": target})
		if err != nil {
			if accountBlame(err) {
				breakerFailWith(name, err)
			}
			log.Printf("web fetch via %s failed: %v", name, err)
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		var out struct {
			Title   string   `json:"title"`
			Content string   `json:"content"`
			Links   []string `json:"links"`
		}
		if err := json.Unmarshal(raw, &out); err != nil {
			if firstErr == nil {
				firstErr = fmt.Errorf("decode %s: %w", name, err)
			}
			continue
		}
		if out.Content == "" && out.Title == "" {
			if firstErr == nil {
				firstErr = fmt.Errorf("%s returned an empty document", name)
			}
			continue
		}
		return fetchDoc{
			URL:     target,
			Status:  http.StatusOK,
			Title:   out.Title,
			Content: out.Content,
			Links:   out.Links,
		}, "ollama:" + name
	}

	// Nothing upstream answered. A refused target is still refused here - the guard runs on
	// the local path, and the vendor does not consult it for us.
	doc := fetchOneFn(ctx, target)
	if doc.Error != "" && firstErr != nil {
		log.Printf("web fetch fell back to the local fetcher after: %v", firstErr)
	}
	return doc, "local"
}

// localSearch is the fallback scraper. A seam, so the fallback can be tested without the
// network: the SSRF guard inside runSearch refuses loopback, and that guard is not a thing a
// test should be allowed to relax.
var localSearch = runSearch

// webBackendLabel reports which path would serve a web call, for operators reading /status.
func webBackendLabel() string {
	names := webUpstreamProviders()
	if len(names) == 0 {
		return "local"
	}
	return "ollama:" + strings.Join(names, ",")
}
