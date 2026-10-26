package main

// OAuth-capable providers.
//
// Nous Portal does not issue long-lived inference keys: it issues a one-hour
// access token minted from a refresh token. A copy of that access token in the
// environment therefore dies within the hour, which is exactly how the nous
// fallback sat dead with a 401 while the account still had credit. Minting from
// the refresh token keeps the provider alive without a human, a cron, or a copy
// of anyone else's credentials.
//
// Verified against the live portal: POST {portal}/api/oauth/token with the
// refresh token in x-nous-refresh-token and grant_type=refresh_token returns a
// new access token, a rotated refresh token, and expires_in=3600. The rotated
// token is used from then on; the previous one stays valid, so a restart that
// falls back to the configured value still works.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

// OAuthConfig describes how to mint an access token for one provider.
type OAuthConfig struct {
	// PortalBase is the origin serving /api/oauth/token.
	PortalBase string
	// ClientID is the public client the refresh token was issued to.
	ClientID string
	// RefreshEnv names the environment variable holding the refresh token.
	RefreshEnv string
}

type tokenState struct {
	access     string
	refresh    string
	expiresAt  time.Time
	lastErr    error
	refreshing bool
}

// oauthSources holds one token source per provider name.
var (
	oauthMu      sync.Mutex
	oauthSources = map[string]*tokenSource{}
)

type tokenSource struct {
	cfg OAuthConfig
	net *http.Client

	mu sync.Mutex
	st tokenState
}

func oauthSourceFor(name string, cfg *OAuthConfig) *tokenSource {
	oauthMu.Lock()
	defer oauthMu.Unlock()
	if ts, ok := oauthSources[name]; ok {
		return ts
	}
	ts := &tokenSource{cfg: *cfg, net: &http.Client{Timeout: 30 * time.Second}}
	oauthSources[name] = ts
	return ts
}

// oauthRefreshMargin renews slightly before expiry: a token that expires mid
// request fails the request, and the upstream treats that as a provider outage.
const oauthRefreshMargin = 5 * time.Minute

// Token returns a usable access token, minting a new one when the cached value is
// missing or about to expire.
func (t *tokenSource) Token(ctx context.Context) (string, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.st.access != "" && time.Now().Before(t.st.expiresAt.Add(-oauthRefreshMargin)) {
		return t.st.access, nil
	}
	if t.st.refresh == "" {
		t.st.refresh = strings.TrimSpace(os.Getenv(t.cfg.RefreshEnv))
	}
	if t.st.refresh == "" {
		return "", fmt.Errorf("no refresh token: set %s", t.cfg.RefreshEnv)
	}
	if err := t.mint(ctx); err != nil {
		t.st.lastErr = err
		return "", err
	}
	return t.st.access, nil
}

func (t *tokenSource) mint(ctx context.Context) error {
	form := url.Values{
		"grant_type": {"refresh_token"},
		"client_id":  {t.cfg.ClientID},
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		strings.TrimRight(t.cfg.PortalBase, "/")+"/api/oauth/token",
		strings.NewReader(form.Encode()))
	if err != nil {
		return fmt.Errorf("build token request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	// The refresh token travels in its own header, not the body: that is the
	// contract the portal accepts, and it keeps it out of any body logging.
	req.Header.Set("x-nous-refresh-token", t.st.refresh)

	resp, err := t.net.Do(req)
	if err != nil {
		return fmt.Errorf("token transport: %w", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != 200 {
		detail := strings.TrimSpace(string(body))
		if len(detail) > 200 {
			detail = detail[:200]
		}
		return fmt.Errorf("token endpoint HTTP %d: %s", resp.StatusCode, detail)
	}

	var out struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		ExpiresIn    int    `json:"expires_in"`
		TokenType    string `json:"token_type"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return fmt.Errorf("token decode: %w", err)
	}
	if out.AccessToken == "" {
		return fmt.Errorf("token endpoint returned no access_token")
	}
	// A rotating refresh token has to be adopted, or the next mint fails with
	// invalid_grant once the old one finally ages out.
	if out.RefreshToken != "" {
		t.st.refresh = out.RefreshToken
	}
	ttl := time.Duration(out.ExpiresIn) * time.Second
	if ttl <= 0 {
		ttl = time.Hour
	}
	t.st.access = out.AccessToken
	t.st.expiresAt = time.Now().Add(ttl)
	t.st.lastErr = nil
	return nil
}

// providerKey returns the bearer token to use for a provider, minting an OAuth
// access token when the provider is configured that way. Every route calls this
// rather than reading the environment directly, so a refreshed credential reaches
// chat, streaming and decision routes alike.
func providerKey(ctx context.Context, name string, p *Provider) (string, error) {
	if p.OAuth != nil {
		return oauthSourceFor(name, p.OAuth).Token(ctx)
	}
	key := os.Getenv(p.KeyEnv)
	if key == "" {
		return "", fmt.Errorf("no api key: set %s", p.KeyEnv)
	}
	return key, nil
}

// providerKeyPresent reports whether a provider has what it needs to authenticate,
// without minting anything: /health must stay cheap and side-effect free.
func providerKeyPresent(name string, p *Provider) bool {
	if p.OAuth != nil {
		if strings.TrimSpace(os.Getenv(p.OAuth.RefreshEnv)) != "" {
			return true
		}
		oauthMu.Lock()
		ts, ok := oauthSources[name]
		oauthMu.Unlock()
		if !ok {
			return false
		}
		ts.mu.Lock()
		defer ts.mu.Unlock()
		return ts.st.access != ""
	}
	return os.Getenv(p.KeyEnv) != ""
}
