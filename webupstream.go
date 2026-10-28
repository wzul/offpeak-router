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
			// The credential and the account are shared with chat, so a failure here is
			// evidence about the account, not only about this route.
			breakerFailWith(name, err)
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
			hits = append(hits, webHit{Title: r.Title, URL: r.URL, Description: r.Content})
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
			breakerFailWith(name, err)
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
