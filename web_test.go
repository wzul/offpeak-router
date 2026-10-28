package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestHandleFetchRefusesInternalTargets(t *testing.T) {
	up := &webRig{}
	up.server(t)
	webRigProviders(t, up.srv.URL)

	_, srv := newUpstream(t)
	router := setup(t, srv)

	// An internal literal is refused before any provider is asked: handing a caller's internal
	// address to a third party is the same mistake as fetching it here.
	for _, target := range []string{"http://169.254.169.254/latest/meta-data/", "http://127.0.0.1:8788/status"} {
		resp, body := postJSON(t, router.URL+"/v1/fetch", `{"url":"`+target+`"}`)
		if resp.StatusCode != 400 {
			t.Errorf("%s: status = %d, want 400; body = %s", target, resp.StatusCode, body)
		}
		var out struct {
			Success *bool `json:"success"`
		}
		if err := json.Unmarshal(body, &out); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if out.Success == nil || *out.Success {
			t.Errorf("%s: success = %v on a refused target", target, out.Success)
		}
	}
	if got := up.recorded(); len(got) != 0 {
		t.Errorf("a refused target was forwarded upstream: %v", got)
	}
}

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

// Refusal must cover host names, not only IP literals: an internal name resolves to an internal
// address, and the account we forward to would happily fetch it.
func TestRefuseTargetCoversNamesAndLiterals(t *testing.T) {
	refused := []string{
		"http://169.254.169.254/latest/meta-data/",
		"http://127.0.0.1:8788/status",
		"http://10.0.0.5/",
		"http://192.168.1.1/",
		"http://172.17.0.1:2375/containers/json",
		"http://[::1]:8788/status",
		"http://localhost:8788/status",
		"http://router.localhost/",
		"http://printer.local/",
		"http://vault.internal/",
		"ftp://example.com/file",
		"http:///path-without-host",
	}
	for _, target := range refused {
		u, err := url.Parse(target)
		if err != nil {
			t.Fatalf("parse %s: %v", target, err)
		}
		if err := refuseTarget(u); err == nil {
			t.Errorf("%s was allowed", target)
		}
	}

	allowed := []string{"https://example.com/", "http://93.184.216.34/", "https://go.dev/doc"}
	for _, target := range allowed {
		u, err := url.Parse(target)
		if err != nil {
			t.Fatalf("parse %s: %v", target, err)
		}
		if err := refuseTarget(u); err != nil {
			t.Errorf("%s was refused: %v", target, err)
		}
	}
}
