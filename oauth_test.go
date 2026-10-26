package main

// A provider whose credential is a one-hour OAuth access token must mint from the
// refresh token, or it is dead most of the day - which is exactly what happened to
// the nous fallback: 401 on every request while the account still had credit.

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func newTestSource(t *testing.T, handler http.HandlerFunc) (*tokenSource, *httptest.Server) {
	t.Helper()
	portal := httptest.NewServer(handler)
	t.Cleanup(portal.Close)

	oauthMu.Lock()
	delete(oauthSources, "nous")
	oauthMu.Unlock()
	t.Cleanup(func() {
		oauthMu.Lock()
		delete(oauthSources, "nous")
		oauthMu.Unlock()
	})

	t.Setenv("NOUS_REFRESH_TOKEN", "refresh-one")
	return oauthSourceFor("nous", &OAuthConfig{
		PortalBase: portal.URL,
		ClientID:   "hermes-cli",
		RefreshEnv: "NOUS_REFRESH_TOKEN",
	}), portal
}

func TestOAuthMintsAndCaches(t *testing.T) {
	var calls int32
	ts, _ := newTestSource(t, func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		if r.URL.Path != "/api/oauth/token" {
			t.Errorf("minted at %q, want /api/oauth/token", r.URL.Path)
		}
		// The refresh token travels in its own header, which is the portal's contract.
		if got := r.Header.Get("x-nous-refresh-token"); got != "refresh-one" {
			t.Errorf("refresh header = %q", got)
		}
		if err := r.ParseForm(); err != nil || r.PostForm.Get("grant_type") != "refresh_token" {
			t.Errorf("grant_type = %q", r.PostForm.Get("grant_type"))
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"access_token":"access-one","refresh_token":"refresh-two","expires_in":3600,"token_type":"Bearer"}`))
	})

	ctx := context.Background()
	for i := 0; i < 3; i++ {
		tok, err := ts.Token(ctx)
		if err != nil {
			t.Fatalf("Token: %v", err)
		}
		if tok != "access-one" {
			t.Fatalf("token = %q", tok)
		}
	}
	if n := atomic.LoadInt32(&calls); n != 1 {
		t.Errorf("minted %d times, want 1: every request would burn the portal", n)
	}
}

// A rotating refresh token that is not adopted makes the next mint fail with
// invalid_grant once the original finally ages out.
func TestOAuthAdoptsRotatedRefreshToken(t *testing.T) {
	var saw []string
	ts, _ := newTestSource(t, func(w http.ResponseWriter, r *http.Request) {
		saw = append(saw, r.Header.Get("x-nous-refresh-token"))
		w.Write([]byte(`{"access_token":"a2","refresh_token":"refresh-rotated","expires_in":3600}`))
	})

	ctx := context.Background()
	if _, err := ts.Token(ctx); err != nil {
		t.Fatal(err)
	}
	// Force the cached access token to look expired.
	ts.mu.Lock()
	ts.st.expiresAt = time.Now().Add(-time.Minute)
	ts.mu.Unlock()

	if _, err := ts.Token(ctx); err != nil {
		t.Fatal(err)
	}
	if len(saw) != 2 {
		t.Fatalf("mints = %d, want 2", len(saw))
	}
	if saw[1] != "refresh-rotated" {
		t.Errorf("second mint sent %q, want the rotated token", saw[1])
	}
}

func TestOAuthRenewsBeforeExpiry(t *testing.T) {
	var calls int32
	ts, _ := newTestSource(t, func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.Write([]byte(`{"access_token":"a","expires_in":3600}`))
	})
	ctx := context.Background()
	if _, err := ts.Token(ctx); err != nil {
		t.Fatal(err)
	}
	// Inside the renewal margin: a token that expires mid-request fails it.
	ts.mu.Lock()
	ts.st.expiresAt = time.Now().Add(time.Minute)
	ts.mu.Unlock()
	if _, err := ts.Token(ctx); err != nil {
		t.Fatal(err)
	}
	if n := atomic.LoadInt32(&calls); n != 2 {
		t.Errorf("mints = %d, want 2, renewal margin ignored", n)
	}
}

func TestOAuthSurfacesPortalError(t *testing.T) {
	ts, _ := newTestSource(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(400)
		w.Write([]byte(`{"error":"invalid_grant","error_description":"Invalid refresh token"}`))
	})
	_, err := ts.Token(context.Background())
	if err == nil {
		t.Fatal("a rejected refresh token must be an error, not an empty key")
	}
	// The cause has to reach the attempt log, or an operator sees "all providers
	// failed" with no idea which credential died.
	for _, want := range []string{"400", "invalid_grant"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

func TestOAuthMissingRefreshToken(t *testing.T) {
	t.Setenv("NOUS_REFRESH_TOKEN", "")
	oauthMu.Lock()
	delete(oauthSources, "nous")
	oauthMu.Unlock()

	ts := oauthSourceFor("nous", &OAuthConfig{
		PortalBase: "https://example.invalid", ClientID: "c", RefreshEnv: "NOUS_REFRESH_TOKEN",
	})
	if _, err := ts.Token(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "NOUS_REFRESH_TOKEN") {
		t.Fatalf("error = %v, want it to name the missing variable", err)
	}
}

// The provider table must point nous at the refresh flow, and /health must report
// presence from the refresh token rather than from a key that no longer exists.
func TestNousProviderUsesOAuth(t *testing.T) {
	prev := provs
	t.Cleanup(func() { provs = prev })
	provs = defaultProviders()

	p := provs["nous"]
	if p.OAuth == nil {
		t.Fatal("nous must be configured for OAuth: a static key expires hourly")
	}
	if p.OAuth.RefreshEnv != "NOUS_REFRESH_TOKEN" || p.OAuth.ClientID != "hermes-cli" {
		t.Errorf("unexpected OAuth config: %+v", *p.OAuth)
	}

	t.Setenv("NOUS_REFRESH_TOKEN", "rt")
	if !providerKeyPresent("nous", p) {
		t.Error("a configured refresh token must count as present")
	}
	t.Setenv("NOUS_REFRESH_TOKEN", "")
	oauthMu.Lock()
	delete(oauthSources, "nous")
	oauthMu.Unlock()
	if providerKeyPresent("nous", p) {
		t.Error("no refresh token and no cached access token must count as absent")
	}
}
