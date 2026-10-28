package main

// The fetch dialer used to commit to the first resolved address. On a host with an IPv6
// address but no IPv6 egress that meant every IPv6-capable site timed out, while sites that
// happened to answer over IPv4 worked - a bug that hid behind a working search path. These
// tests hold the fix in place: a dead family must be abandoned in favour of a live one, and
// the SSRF guard must still decide what may be dialled at all.

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

type fakeConn struct{ net.Conn }

func TestDialRaceFallsBackToTheLiveAddress(t *testing.T) {
	dead := net.ParseIP("2001:db8::1")
	live := net.ParseIP("93.184.216.34")

	var tried []string
	dial := func(ctx context.Context, ip net.IP) (net.Conn, error) {
		tried = append(tried, ip.String())
		if ip.Equal(dead) {
			// Models the real failure: the address resolves and nothing ever answers.
			<-ctx.Done()
			return nil, ctx.Err()
		}
		return fakeConn{}, nil
	}

	start := time.Now()
	conn, err := dialRace(context.Background(), []net.IP{dead, live}, dial)
	if err != nil {
		t.Fatalf("dialRace: %v", err)
	}
	if conn == nil {
		t.Fatal("dialRace returned no connection")
	}
	elapsed := time.Since(start)
	if elapsed > 3*time.Second {
		t.Errorf("fallback took %s; a dead family must not cost the whole timeout", elapsed)
	}
	if len(tried) != 2 {
		t.Errorf("tried %v, want both addresses", tried)
	}
}

func TestDialRaceReportsFailureWhenNothingAnswers(t *testing.T) {
	dial := func(ctx context.Context, ip net.IP) (net.Conn, error) {
		return nil, errors.New("connection refused")
	}
	_, err := dialRace(context.Background(), []net.IP{
		net.ParseIP("93.184.216.34"), net.ParseIP("1.1.1.1"),
	}, dial)
	if err == nil {
		t.Fatal("dialRace reported success when every address failed")
	}
	// The first real error is the useful one; replacing it with a generic message throws
	// away the only clue about why the host is unreachable.
	if !strings.Contains(err.Error(), "connection refused") {
		t.Errorf("error = %q, want the underlying dial error", err)
	}
}

// The guard must still be the thing that decides, even now that several addresses are tried.
func TestDialCandidatesKeepsOnlyPublicAddresses(t *testing.T) {
	ips := []net.IPAddr{
		{IP: net.ParseIP("127.0.0.1")},
		{IP: net.ParseIP("169.254.169.254")},
		{IP: net.ParseIP("172.17.0.1")},
		{IP: net.ParseIP("93.184.216.34")},
	}
	got := dialCandidates(ips)
	if len(got) != 1 || !got[0].Equal(net.ParseIP("93.184.216.34")) {
		t.Fatalf("dialCandidates = %v, want only the public address", got)
	}
	if got := dialCandidates([]net.IPAddr{{IP: net.ParseIP("10.0.0.1")}}); len(got) != 0 {
		t.Errorf("a private-only host must yield no candidates, got %v", got)
	}
}

// An end-to-end fetch test cannot live here: the guard correctly refuses loopback, so the
// only addresses a test could answer on are the ones the guard blocks. The live proof for
// this is a real fetch through the deployed router; these tests pin the fallback logic.

// safeDialContext must try every resolved address, not commit to the first. This is the
// regression that broke every IPv6-capable fetch on a host without IPv6 egress.
func TestSafeDialContextTriesEveryResolvedAddress(t *testing.T) {
	prevLookup, prevDial := lookupIPAddrs, webDialFn
	t.Cleanup(func() { lookupIPAddrs, webDialFn = prevLookup, prevDial })

	dead := net.ParseIP("2001:db8::1")
	live := net.ParseIP("93.184.216.34")
	lookupIPAddrs = func(ctx context.Context, host string) ([]net.IPAddr, error) {
		return []net.IPAddr{{IP: dead}, {IP: live}}, nil
	}

	var tried []string
	webDialFn = func(ctx context.Context, network string, ip net.IP, port string) (net.Conn, error) {
		tried = append(tried, ip.String()+":"+port)
		if ip.Equal(dead) {
			<-ctx.Done()
			return nil, ctx.Err()
		}
		return fakeConn{}, nil
	}

	// A bounded context on purpose: if the dialer ever commits to the dead address again,
	// this must fail in seconds, not hang until the package timeout.
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	conn, err := safeDialContext(ctx, "tcp", "example.com:443")
	if err != nil {
		t.Fatalf("safeDialContext: %v", err)
	}
	if conn == nil {
		t.Fatal("no connection")
	}
	if len(tried) != 2 {
		t.Errorf("tried %v, want the dead address and then the live one", tried)
	}
	if want := "93.184.216.34:443"; tried[len(tried)-1] != want {
		t.Errorf("last attempt = %q, want %q", tried[len(tried)-1], want)
	}
}

// A fetch that failed must not be reported as a successful empty page. Every HTTP client
// reads the status code first; a 200 here means the caller keeps an empty document and never
// learns why.
func TestFetchRouteReportsFailure(t *testing.T) {
	prevFetch := fetchForModel
	t.Cleanup(func() { fetchForModel = prevFetch })
	fetchForModel = func(ctx context.Context, target string) fetchDoc {
		if strings.Contains(target, "hidup") {
			return fetchDoc{URL: target, Status: 200, Title: "Hidup", Content: "kandungan"}
		}
		return fetchDoc{URL: target, Error: "dial tcp: i/o timeout"}
	}
	// The route calls fetchOne directly, so the seam is replaced for this test.
	prevOne := fetchOneFn
	t.Cleanup(func() { fetchOneFn = prevOne })
	fetchOneFn = fetchForModel

	router := httptest.NewServer(newMux())
	t.Cleanup(router.Close)

	// Total failure: 502, never success:true.
	resp, raw := postAnthropic(t, router.URL, "/v1/fetch", `{"url":"https://mati.example/"}`)
	if resp.StatusCode != 502 {
		t.Errorf("all-failed fetch returned HTTP %d, want 502: %s", resp.StatusCode, raw)
	}
	var gone struct {
		Success *bool `json:"success"`
		Data    []struct {
			Error string `json:"error"`
		} `json:"data"`
	}
	if err := json.Unmarshal([]byte(raw), &gone); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if gone.Success == nil || *gone.Success {
		t.Errorf("success = %v on a total failure", gone.Success)
	}
	if len(gone.Data) != 1 || gone.Data[0].Error == "" {
		t.Errorf("the per-URL error was dropped: %s", raw)
	}

	// Partial: 200, but success is still false.
	resp, raw = postAnthropic(t, router.URL, "/v1/fetch",
		`{"urls":["https://hidup.example/","https://mati.example/"]}`)
	if resp.StatusCode != 200 {
		t.Errorf("partial fetch returned HTTP %d, want 200", resp.StatusCode)
	}
	var part struct {
		Success *bool `json:"success"`
		Failed  int   `json:"failed"`
	}
	if err := json.Unmarshal([]byte(raw), &part); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if part.Success == nil || *part.Success {
		t.Error("partial failure reported success:true")
	}
	if part.Failed != 1 {
		t.Errorf("failed = %d, want 1", part.Failed)
	}

	// Clean: 200 and success:true.
	resp, raw = postAnthropic(t, router.URL, "/v1/fetch", `{"url":"https://hidup.example/"}`)
	if resp.StatusCode != 200 {
		t.Errorf("good fetch returned HTTP %d", resp.StatusCode)
	}
	if !strings.Contains(string(raw), `"success":true`) {
		t.Errorf("a good fetch should stay success:true: %s", raw)
	}
}
