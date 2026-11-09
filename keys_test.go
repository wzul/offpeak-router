package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestParseKeysAcceptsLabelTokenAndLimit(t *testing.T) {
	got, bad := parseKeys("fleet:tok-a, editor:tok-b:250000 ,")
	if len(bad) != 0 {
		t.Fatalf("unexpected malformed entries: %v", bad)
	}
	if len(got) != 2 {
		t.Fatalf("want 2 keys, got %d", len(got))
	}
	if got[0].Label != "fleet" || got[0].Token != "tok-a" || got[0].MonthlyTokenLimit != 0 {
		t.Fatalf("first key wrong: %+v", got[0])
	}
	if got[1].Label != "editor" || got[1].Token != "tok-b" || got[1].MonthlyTokenLimit != 250000 {
		t.Fatalf("second key wrong: %+v", got[1])
	}
}

func TestParseKeysReportsMalformedEntries(t *testing.T) {
	_, bad := parseKeys("nokey, :no-label, bad:tok:notanumber, ok:tok2")
	if len(bad) != 3 {
		t.Fatalf("want 3 malformed entries, got %d: %v", len(bad), bad)
	}
}

// A key must be recognised under its own label, and an unknown token must not
// be accepted just because some other key exists.
func TestResolveKeyMapsTokenToLabel(t *testing.T) {
	defer func(prev []APIKey, prevTok string) { keys, authToken = prev, prevTok }(keys, authToken)
	keys = []APIKey{{Label: "fleet", Token: "tok-a"}, {Label: "editor", Token: "tok-b"}}
	authToken = "tok-a"

	if label, ok := resolveKey("tok-a"); !ok || label != "fleet" {
		t.Fatalf("tok-a → %q,%v; want fleet,true", label, ok)
	}
	if label, ok := resolveKey("tok-b"); !ok || label != "editor" {
		t.Fatalf("tok-b → %q,%v; want editor,true", label, ok)
	}
	if _, ok := resolveKey("tok-c"); ok {
		t.Fatal("unknown token accepted")
	}
	if _, ok := resolveKey(""); ok {
		t.Fatal("empty token accepted")
	}
}

// The legacy single-token deployment keeps working, filed under "default".
func TestResolveKeyFallsBackToLegacyToken(t *testing.T) {
	defer func(prev []APIKey, prevTok string) { keys, authToken = prev, prevTok }(keys, authToken)
	keys = nil
	authToken = "legacy"

	if label, ok := resolveKey("legacy"); !ok || label != "default" {
		t.Fatalf("legacy token → %q,%v; want default,true", label, ok)
	}
	if _, ok := resolveKey("nope"); ok {
		t.Fatal("wrong token accepted")
	}

	authToken = ""
	if _, ok := resolveKey(""); !ok {
		t.Fatal("auth should be disabled when no keys and no legacy token")
	}
}

// withAuth must put the label in the request context so the recording paths
// file tokens under the right key, and must 401 everything else.
func TestWithAuthPropagatesLabel(t *testing.T) {
	defer func(prev []APIKey, prevTok string) { keys, authToken = prev, prevTok }(keys, authToken)
	keys = []APIKey{{Label: "fleet", Token: "tok-a"}}
	authToken = "tok-a"

	var label string
	handler := withAuth(func(w http.ResponseWriter, r *http.Request) {
		label = labelFromRequest(r)
		w.WriteHeader(200)
	})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer tok-a")
	handler(rec, req)
	if label != "fleet" {
		t.Fatalf("label in context = %q; want fleet", label)
	}
	if rec.Code != 200 {
		t.Fatalf("good token status = %d; want 200", rec.Code)
	}

	rec = httptest.NewRecorder()
	req = httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer wrong")
	handler(rec, req)
	if rec.Code != 401 {
		t.Fatalf("bad token status = %d; want 401", rec.Code)
	}

	// The Anthropic dialect sends x-api-key instead of Authorization.
	label = ""
	rec = httptest.NewRecorder()
	req = httptest.NewRequest("POST", "/v1/messages", nil)
	req.Header.Set("x-api-key", "tok-a")
	handler(rec, req)
	if rec.Code != 200 || label != "fleet" {
		t.Fatalf("x-api-key: status %d label %q; want 200 fleet", rec.Code, label)
	}
}
