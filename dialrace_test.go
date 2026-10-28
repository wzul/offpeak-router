package main

// The fetch dialer used to commit to the first resolved address. On a host with an IPv6
// address but no IPv6 egress that meant every IPv6-capable site timed out, while sites that
// happened to answer over IPv4 worked - a bug that hid behind a working search path. These
// tests hold the fix in place: a dead family must be abandoned in favour of a live one, and
// the SSRF guard must still decide what may be dialled at all.

import (
	"context"
	"errors"
	"net"
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
