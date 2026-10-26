package main

// Upstream identity headers.
//
// OpenCode Go rejects a request with no x-opencode-session, so the router already
// presents itself as that client. Presenting only half of it was worse than not
// trying: a User-Agent reading "offpeak-router/1.9.0" beside a session id prefixed
// "offpeak-" told the provider exactly which third-party proxy was calling, while
// looking like a client that had been rewritten by hand.
//
// The values below were measured, not guessed: a throwaway OpenAI-compatible
// endpoint was stood up and the installed CLI (opencode v2.1.2) was pointed at it
// through an injected config, and its request headers were recorded. When that
// client moves, re-measure and bump OpenCodeUserAgent — or set ROUTER_UA.

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math/big"
	"net/http"
	"os"
	"strings"
	"sync"
)

// OpenCodeUserAgent is the User-Agent the opencode CLI sends. It replaces a
// product name that identified this router to every upstream.
const OpenCodeUserAgent = "opencode/latest/2.1.2/cli"

// DefaultUserAgent is what providers get when they do not name their own. The old
// value also existed because Cloudflare returns 403 to Go's default UA; that still
// holds for any identifiable UA.
var DefaultUserAgent = OpenCodeUserAgent

// opencodeClientHeader is the companion header the CLI sends with the session id.
const opencodeClientHeader = "cli"

// projectID is stable for the lifetime of the process, the way a project hash is
// stable for a working directory. Two requests hours apart look like the same
// client rather than a new one every call.
var projectID = func() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return strings.Repeat("0", 32)
	}
	return hex.EncodeToString(b)
}()

// base62 encodes bytes the way an id generator would: no separators, no padding,
// fixed length for a fixed input.
func base62(b []byte, width int) string {
	const alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	n := new(big.Int).SetBytes(b)
	base := big.NewInt(62)
	mod := new(big.Int)
	out := make([]byte, 0, width)
	for n.Sign() > 0 {
		n.DivMod(n, base, mod)
		out = append(out, alphabet[mod.Int64()])
	}
	for len(out) < width {
		out = append(out, '0')
	}
	if len(out) > width {
		out = out[len(out)-width:]
	}
	// Reverse: DivMod yields least-significant digit first.
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return string(out)
}

// sha256Bytes is the digest behind a conversation id.
func sha256Bytes(s string) []byte {
	sum := sha256.Sum256([]byte(s))
	return sum[:]
}

// opencodeSessionID is the id shape the CLI uses: "ses_" plus 24 mixed-case
// characters.
func opencodeSessionID(seed string) string {
	sum := sha256Bytes(seed)
	return "ses_" + base62(sum[:18], 24)
}

var (
	traceMu   sync.Mutex
	traceBase = func() [16]byte {
		var b [16]byte
		_, _ = rand.Read(b[:])
		return b
	}()
)

// newTraceHeaders returns a W3C traceparent plus the Zipkin b3 line the CLI sends
// alongside it. A client that carries a session id but no trace context is another
// inconsistency worth removing.
func newTraceHeaders() (traceparent, b3 string) {
	traceMu.Lock()
	// Keep the trace id constant for a short burst, like one trace per request
	// chain, and vary the span id per call.
	span := make([]byte, 8)
	_, _ = rand.Read(span)
	parent := make([]byte, 8)
	_, _ = rand.Read(parent)
	traceMu.Unlock()

	traceID := hex.EncodeToString(traceBase[:])
	spanID := hex.EncodeToString(span)
	parentID := hex.EncodeToString(parent)
	traceparent = fmt.Sprintf("00-%s-%s-01", traceID, spanID)
	b3 = fmt.Sprintf("%s-%s-1-%s", traceID, spanID, parentID)
	return traceparent, b3
}

// applyUpstreamHeaders sets every header the upstream sees. All three route
// families call this: a fix applied to one handler is a fix the other two never
// get.
func applyUpstreamHeaders(req *http.Request, p *Provider, r *http.Request, body map[string]any) {
	ua := p.UA
	if ua == "" {
		ua = strings.TrimSpace(os.Getenv("ROUTER_UA"))
	}
	if ua == "" {
		ua = DefaultUserAgent
	}
	req.Header.Set("User-Agent", ua)

	if !p.SessionHeader {
		return
	}

	session := conversationID(r, body)
	req.Header.Set("x-opencode-session", session)
	// The CLI sends the same id under four names; a provider or an observability
	// pipeline may key on any of them.
	req.Header.Set("x-opencode-session-id", session)
	req.Header.Set("x-session-affinity", session)
	req.Header.Set("x-session-id", session)
	req.Header.Set("x-opencode-client", opencodeClientHeader)
	req.Header.Set("x-opencode-project", projectID)
	traceparent, b3 := newTraceHeaders()
	req.Header.Set("traceparent", traceparent)
	req.Header.Set("b3", b3)
}
