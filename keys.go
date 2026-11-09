package main

import (
	"context"
	"crypto/subtle"
	"net/http"
	"os"
	"strconv"
	"strings"
)

// APIKey is one bearer token, plus the label its usage is filed under.
//
// Several keys exist because several clients share this router and their spend
// has to be attributable: the fleet bot, an editor on a laptop, a release
// script. One shared token can only ever answer "the router used this much".
type APIKey struct {
	Label string
	Token string
	// MonthlyTokenLimit is reported by /v1/usage and enforced on chat routes
	// when greater than zero. Zero means unlimited.
	MonthlyTokenLimit int64
}

var (
	keys []APIKey
)

type ctxKey struct{}

// parseKeys reads `label:token[:monthly_limit]` entries separated by commas.
// Whitespace around entries is ignored, and a malformed entry is reported so
// a typo does not silently disable a client.
func parseKeys(spec string) ([]APIKey, []string) {
	var out []APIKey
	var bad []string
	for _, raw := range strings.Split(spec, ",") {
		entry := strings.TrimSpace(raw)
		if entry == "" {
			continue
		}
		parts := strings.Split(entry, ":")
		if len(parts) < 2 || strings.TrimSpace(parts[0]) == "" || strings.TrimSpace(parts[1]) == "" {
			bad = append(bad, raw)
			continue
		}
		k := APIKey{Label: strings.TrimSpace(parts[0]), Token: strings.TrimSpace(parts[1])}
		if len(parts) >= 3 && strings.TrimSpace(parts[2]) != "" {
			n, err := strconv.ParseInt(strings.TrimSpace(parts[2]), 10, 64)
			if err != nil || n < 0 {
				bad = append(bad, raw)
				continue
			}
			k.MonthlyTokenLimit = n
		}
		out = append(out, k)
	}
	return out, bad
}

// loadKeys builds the key table from ROUTER_KEYS, then folds in the legacy
// ROUTER_AUTH_TOKEN so an existing deployment keeps working untouched.
func loadKeys() []string {
	spec := os.Getenv("ROUTER_KEYS")
	parsed, bad := parseKeys(spec)
	if legacy := os.Getenv("ROUTER_AUTH_TOKEN"); legacy != "" {
		label := os.Getenv("ROUTER_KEY_LABEL")
		if label == "" {
			label = "default"
		}
		parsed = append(parsed, APIKey{Label: label, Token: legacy})
	}
	// Keep the legacy field in step: other code paths still read authToken to
	// decide whether auth is on at all.
	if len(parsed) > 0 && authToken == "" {
		authToken = parsed[0].Token
	}
	keys = parsed
	return bad
}

// resolveKey returns the label that owns token. When no key table has been
// loaded, the legacy single ROUTER_AUTH_TOKEN still decides, under the label
// "default"; with neither, auth is off and every caller is anonymous.
func resolveKey(token string) (string, bool) {
	if len(keys) == 0 {
		if authToken == "" {
			return "", true
		}
		if subtle.ConstantTimeCompare([]byte(token), []byte(authToken)) == 1 {
			return "default", true
		}
		return "", false
	}
	label := ""
	found := false
	// Compare against every key, without an early break: stopping at the first
	// match would leak, by timing, a token's position in the table.
	for _, k := range keys {
		if subtle.ConstantTimeCompare([]byte(token), []byte(k.Token)) == 1 {
			if !found {
				label, found = k.Label, true
			}
		}
	}
	return label, found
}

// tokenFromRequest reads the credential from either header shape: OpenAI-shaped
// clients send `Authorization: Bearer …`, the Anthropic dialect sends
// `x-api-key`, and a client written against one sends nothing else.
func tokenFromRequest(r *http.Request) string {
	if key := r.Header.Get("x-api-key"); key != "" {
		return key
	}
	const prefix = "Bearer "
	if h := r.Header.Get("Authorization"); strings.HasPrefix(h, prefix) {
		return strings.TrimPrefix(h, prefix)
	}
	return ""
}

// labelFromRequest is how the recording paths learn whose usage a row belongs
// to. Handlers are wrapped by withAuth, which puts the label in the context.
func labelFromRequest(r *http.Request) string {
	if r == nil {
		return ""
	}
	if v, ok := r.Context().Value(ctxKey{}).(string); ok {
		return v
	}
	return ""
}

func withLabel(r *http.Request, label string) *http.Request {
	return r.WithContext(context.WithValue(r.Context(), ctxKey{}, label))
}
