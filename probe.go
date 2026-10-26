package main

// Holding a provider out indefinitely, without ever giving up on it.
//
// A 60-second cooldown is right for a hiccup and wrong for an unpaid bill: a bill stays
// unpaid for hours, so the router would waste a call every minute rediscovering it, and a
// key that comes back to life would be ignored for a minute after it does. Those failures
// are therefore held until a probe says otherwise - and the probe runs here, in this process,
// on its own schedule, so nothing depends on traffic arriving to notice a recovery.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"time"
)

// upstreamError keeps the status code, which is what tells a dead account from a bad minute.
// A formatted string would have to be parsed back, and parsing your own error messages is a
// bug waiting to be reintroduced.
type upstreamError struct {
	Status int
	Detail string
}

func (e *upstreamError) Error() string {
	return fmt.Sprintf("HTTP %d: %s", e.Status, e.Detail)
}

func newUpstreamError(status int, detail string) error {
	return &upstreamError{Status: status, Detail: strings.TrimSpace(detail)}
}

// billingWords are what upstreams say when the account, not the request, is the problem.
var billingWords = []string{
	"past due", "payment", "billing", "subscription", "quota", "credit",
	"insufficient", "exhausted", "not active", "suspended",
}

// holdsIndefinitely reports whether this failure is the kind that will not clear in a
// minute: no money, no valid key, no allowance.
func holdsIndefinitely(err error) (bool, string) {
	var up *upstreamError
	if !errors.As(err, &up) {
		return false, ""
	}
	switch up.Status {
	case http.StatusUnauthorized: // 401: the key is wrong or revoked
		return true, "unauthorized"
	case http.StatusPaymentRequired: // 402
		return true, "payment required"
	case http.StatusForbidden: // 403: sometimes a bill, sometimes not
		detail := strings.ToLower(up.Detail)
		for _, w := range billingWords {
			if strings.Contains(detail, w) {
				return true, "forbidden: " + w
			}
		}
		return false, ""
	}
	return false, ""
}

// Probe schedule: first look after a minute, then back off to hourly. Fast enough that a
// topped-up account is picked up quickly, slow enough that a long dead one costs nothing.
const (
	probeFirstGap = 60 * time.Second
	probeMaxGap   = time.Hour
	probeInterval = 15 * time.Second
)

// probeProviderFn is what a probe actually does. A variable so tests drive it directly:
// probing a real provider on a timer cannot be part of a repeatable test.
var probeProviderFn = probeProviderHTTP

// breakerHold parks a provider until a probe succeeds.
func breakerHold(p, reason string) {
	brMu.Lock()
	defer brMu.Unlock()
	e := breakers[p]
	if e == nil {
		e = &breakerEntry{}
		breakers[p] = e
	}
	if !e.held {
		log.Printf("provider %s held out (%s); probing every %s until it recovers",
			p, reason, probeFirstGap)
		e.probeGap = probeFirstGap
	}
	e.held = true
	e.openUntil = time.Time{}
	e.nextProbe = nowFn().Add(e.probeGap)
}

// breakerFailWith records a failure and picks the right kind of penalty.
func breakerFailWith(p string, err error) {
	if hold, reason := holdsIndefinitely(err); hold {
		breakerHold(p, reason)
		return
	}
	breakerFail(p)
}

// breakerProbeDue lists the held providers whose next look is due.
func breakerProbeDue(now time.Time) []string {
	brMu.Lock()
	defer brMu.Unlock()
	var due []string
	for p, e := range breakers {
		if e.held && !now.Before(e.nextProbe) {
			due = append(due, p)
		}
	}
	return due
}

// breakerProbeResult applies one probe outcome: a working provider is released at once, a
// failing one waits longer.
func breakerProbeResult(p string, ok bool) {
	brMu.Lock()
	defer brMu.Unlock()
	e := breakers[p]
	if e == nil {
		return
	}
	if ok {
		log.Printf("provider %s answered a probe; available again", p)
		breakers[p] = &breakerEntry{}
		return
	}
	e.probeGap *= 2
	if e.probeGap > probeMaxGap {
		e.probeGap = probeMaxGap
	}
	e.nextProbe = nowFn().Add(e.probeGap)
	log.Printf("provider %s still unavailable; next probe in %s", p, e.probeGap)
}

// startProber runs the probe loop for the life of the process.
func startProber(ctx context.Context) {
	go func() {
		t := time.NewTicker(probeInterval)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				runProbeRound(ctx)
			}
		}
	}()
}

// runProbeRound probes every held provider that is due. Split out so a test can run exactly
// one round instead of sleeping.
func runProbeRound(ctx context.Context) {
	for _, name := range breakerProbeDue(nowFn()) {
		p, ok := provs[name]
		if !ok {
			// The provider is gone from the config: stop holding a name that cannot come back.
			breakerProbeResult(name, true)
			continue
		}
		breakerProbeResult(name, probeProviderFn(ctx, name, p))
	}
}

// probeProviderHTTP makes the cheapest request that answers the question: does this account
// work right now? One token on a chat model the provider serves.
func probeProviderHTTP(ctx context.Context, name string, p *Provider) bool {
	name2, route, ok := probeableModel(p)
	if !ok {
		log.Printf("provider %s: no chat model to probe with; staying held", name)
		return false
	}
	key, err := providerKey(ctx, name, p)
	if err != nil {
		// The credential is the thing being tested: an unresolvable key is not an answer,
		// it is the same failure, so stay held and try again later.
		return false
	}

	_ = name2 // the canonical name is not needed: only the provider's own id is sent
	body := map[string]any{
		"model":      route.ID,
		"max_tokens": 1,
		"messages":   []any{map[string]any{"role": "user", "content": "ping"}},
		"stream":     false,
	}
	buf, err := json.Marshal(body)
	if err != nil {
		return false
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.Base+"/chat/completions", strings.NewReader(string(buf)))
	if err != nil {
		return false
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	// A probe has no client request. headers.go walks the client request for the session
	// id, so hand it an empty one rather than a nil it would dereference.
	applyUpstreamHeaders(req, p, httptest.NewRequest(http.MethodPost, "/probe", nil), body)

	resp, err := httpc.Do(req)
	if err != nil {
		log.Printf("provider %s probe: %v", name, err)
		return false
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, io.LimitReader(resp.Body, 2048))
	if resp.StatusCode >= 400 {
		log.Printf("provider %s probe: HTTP %d", name, resp.StatusCode)
		return false
	}
	return true
}

// probeableModel picks a chat model to probe with. The canonical one wins, so the probe
// exercises the path real traffic takes.
func probeableModel(p *Provider) (string, ModelRoute, bool) {
	if route, ok := p.Models[canonicalModel]; ok && modelAPIIsChat(route) {
		return canonicalModel, route, true
	}
	for name, route := range p.Models {
		if modelAPIIsChat(route) {
			return name, route, true
		}
	}
	if len(p.Models) == 0 && p.Model != "" {
		return p.Model, ModelRoute{ID: p.Model}, true
	}
	return "", ModelRoute{}, false
}

func modelAPIIsChat(route ModelRoute) bool {
	return route.API == "" || route.API == APIChat
}
