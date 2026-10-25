package main

import (
	"os"
	"testing"
	"time"
)

// candidateOrder works over whatever providers are loaded, so the policy
// tests need the built-in set in place.
func TestMain(m *testing.M) {
	if provs == nil {
		provs = defaultProviders()
	}
	os.Exit(m.Run())
}

func mon(hour int) time.Time {
	return time.Date(2026, 10, 19, hour, 0, 0, 0, time.UTC) // Monday
}

func sat(hour int) time.Time {
	return time.Date(2026, 10, 24, hour, 0, 0, 0, time.UTC)
}

func zero() map[string]float64 {
	return map[string]float64{"go": 0, "ollama": 0, "nous": 0}
}

func TestPeakBoundariesGo(t *testing.T) {
	cases := []struct {
		hour int
		want bool
	}{
		{0, false}, {1, true}, {2, true}, {3, true}, {4, false}, {5, false},
		{6, true}, {7, true}, {8, true}, {9, true}, {10, false}, {11, false},
		{12, false}, {17, false}, {18, false}, {23, false},
	}
	for _, c := range cases {
		if got := isPeak("go", mon(c.hour)); got != c.want {
			t.Errorf("go peak @ %02d:00 = %v, want %v", c.hour, got, c.want)
		}
	}
	// half-open: 03:59 in, 04:00 out
	if !isPeak("go", mon(3).Add(59*time.Minute)) {
		t.Error("go 03:59 should be peak")
	}
	if isPeak("go", mon(4)) {
		t.Error("go 04:00 should be off-peak")
	}
}

func TestPeakBoundariesOllama(t *testing.T) {
	cases := []struct {
		hour int
		want bool
	}{
		{0, false}, {1, false}, {6, false}, {10, false}, {11, false},
		{12, true}, {13, true}, {17, true}, {18, false}, {23, false},
	}
	for _, c := range cases {
		if got := isPeak("ollama", mon(c.hour)); got != c.want {
			t.Errorf("ollama peak @ %02d:00 = %v, want %v", c.hour, got, c.want)
		}
	}
	if !isPeak("ollama", mon(17).Add(59*time.Minute)) {
		t.Error("ollama 17:59 should be peak")
	}
	if isPeak("ollama", mon(18)) {
		t.Error("ollama 18:00 should be off-peak")
	}
}

func TestWeekendAllOffPeak(t *testing.T) {
	for h := 0; h < 24; h++ {
		if isPeak("go", sat(h)) {
			t.Errorf("go peak on Saturday %02d:00", h)
		}
		if isPeak("ollama", sat(h)) {
			t.Errorf("ollama peak on Saturday %02d:00", h)
		}
	}
}

func TestNousNeverPeaks(t *testing.T) {
	for h := 0; h < 24; h++ {
		if isPeak("nous", mon(h)) {
			t.Errorf("nous peaked at %02d:00", h)
		}
	}
}

// TestPeakWindowsDisjoint is the invariant the whole design rests on: if the
// windows ever overlapped, some hour would have no off-peak provider.
func TestPeakWindowsDisjoint(t *testing.T) {
	for h := 0; h < 24; h++ {
		if isPeak("go", mon(h)) && isPeak("ollama", mon(h)) {
			t.Fatalf("both providers peak at %02d:00", h)
		}
	}
	for h := 0; h < 24; h++ {
		if !isPeak("go", mon(h)) || !isPeak("ollama", mon(h)) {
			continue
		}
		t.Fatalf("no off-peak provider at %02d:00", h)
	}
}

func TestPeakIsDoubleOffPeak(t *testing.T) {
	for _, p := range []string{"go", "ollama"} {
		off := requestCost(p, 6801, 170536, 880, false)
		pk := requestCost(p, 6801, 170536, 880, true)
		if diff := pk - 2*off; diff > 1e-12 || diff < -1e-12 {
			t.Errorf("%s peak %.9f != 2x off-peak %.9f", p, pk, off)
		}
	}
}

func TestNousCheaperOnlyAtOllamaPeak(t *testing.T) {
	ollOff := requestCost("ollama", 6801, 170536, 880, false)
	ollPk := requestCost("ollama", 6801, 170536, 880, true)
	nous := requestCost("nous", 6801, 170536, 880, false)

	if !(ollOff < nous) {
		t.Error("ollama off-peak should be cheaper than nous")
	}
	if !(nous < ollPk) {
		t.Error("nous should be cheaper than ollama at peak")
	}
}

func TestCandidateOrder(t *testing.T) {
	fresh := zero()
	goDead := map[string]float64{"go": 80, "ollama": 0, "nous": 0}
	bothDead := map[string]float64{"go": 80, "ollama": 60, "nous": 0}

	cases := []struct {
		name  string
		at    time.Time
		spent map[string]float64
		want  string
	}{
		{"both off-peak -> go", mon(0), fresh, "go"},
		{"11:00 both off -> go", mon(11), fresh, "go"},
		{"go peak -> ollama", mon(2), fresh, "ollama"},
		{"08:00 go peak -> ollama", mon(8), fresh, "ollama"},
		{"ollama peak -> go", mon(13), fresh, "go"},
		{"17:00 ollama peak -> go", mon(17), fresh, "go"},
		{"go dead at 02:00 -> ollama", mon(2), goDead, "ollama"},
		{"go dead at 00:00 -> ollama", mon(0), goDead, "ollama"},
		{"both dead 02:00 -> ollama topup", mon(2), bothDead, "ollama"},
		{"both dead 13:00 -> nous", mon(13), bothDead, "nous"},
	}
	for _, c := range cases {
		got := candidateOrder(c.at, c.spent)
		if got[0] != c.want {
			t.Errorf("%s: first = %q, want %q (order %v)", c.name, got[0], c.want, got)
		}
	}
}

func TestOrderAlwaysComplete(t *testing.T) {
	fresh := zero()
	for h := 0; h < 24; h++ {
		order := candidateOrder(mon(h), fresh)
		seen := map[string]bool{}
		for _, p := range order {
			if seen[p] {
				t.Errorf("%02d:00 duplicate provider %q", h, p)
			}
			seen[p] = true
		}
		for _, p := range []string{"go", "ollama", "nous"} {
			if !seen[p] {
				t.Errorf("%02d:00 missing provider %q", h, p)
			}
		}
	}
}

// A hard-capped provider must vanish from the order once its allowance is
// spent: it cannot serve at any price, so listing it only wastes a round-trip.
func TestHardCappedProviderDropped(t *testing.T) {
	spent := map[string]float64{"go": 80, "ollama": 0, "nous": 0}
	for h := 0; h < 24; h++ {
		for _, p := range candidateOrder(mon(h), spent) {
			if p == "go" {
				t.Fatalf("%02d:00: exhausted hard-capped go still in the order", h)
			}
		}
	}
}

// A provider past its allowance that CAN be topped up stays in the chain.
func TestTopupProviderSurvives(t *testing.T) {
	spent := map[string]float64{"go": 80, "ollama": 60, "nous": 0}
	found := false
	for _, p := range candidateOrder(mon(2), spent) {
		if p == "ollama" {
			found = true
		}
	}
	if !found {
		t.Fatal("exhausted but top-up-able ollama was dropped from the order")
	}
}

// candidateOrder must not know any provider by name: a config file can load an
// arbitrary set and the tiers still have to work.
func TestArbitraryProviderSet(t *testing.T) {
	savedProvs, savedWindows := provs, peakWindows
	savedAllowance, savedRates, savedCap := allowance, rates, hardCap
	defer func() {
		provs, peakWindows = savedProvs, savedWindows
		allowance, rates, hardCap = savedAllowance, savedRates, savedCap
	}()
	peakWindows = map[string][]Window{}
	allowance = map[string]float64{}
	rates = map[string]Rates{}
	hardCap = map[string]bool{}

	provs = map[string]*Provider{
		"alpha": {Base: "b", Model: "m", KeyEnv: "K1"},
		"beta":  {Base: "b", Model: "m", KeyEnv: "K2"},
	}
	// alpha: off-peak now (no windows), allowance left.
	// beta:  peaks 00-24 would be invalid, so give it a window that misses 02:00.
	peakWindows["alpha"] = nil
	peakWindows["beta"] = []Window{{6, 10}}
	allowance["alpha"] = 10
	allowance["beta"] = 10
	rates["alpha"] = Rates{0.10, 0.001, 0.40, 2.0}
	rates["beta"] = Rates{0.20, 0.002, 0.80, 2.0}

	order := candidateOrder(mon(2), map[string]float64{"alpha": 0, "beta": 0})
	if len(order) != 2 {
		t.Fatalf("order = %v, want both providers", order)
	}
	// both off-peak at 02:00, alpha is cheaper -> alpha first
	if order[0] != "alpha" {
		t.Errorf("order[0] = %q, want alpha (cheaper off-peak)", order[0])
	}
	// names appearing in the default failover chain but not loaded must not leak
	for _, p := range order {
		if p != "alpha" && p != "beta" {
			t.Errorf("leaked unknown provider %q into the order", p)
		}
	}
}

func TestRemaining(t *testing.T) {
	spent := map[string]float64{"go": 80, "ollama": 30, "nous": 0}
	if r, capped := remaining("go", spent); !capped || r != 0 {
		t.Errorf("go should be exhausted, got %.2f capped=%v", r, capped)
	}
	if r, capped := remaining("ollama", spent); !capped || r != 30 {
		t.Errorf("ollama should have 30 left, got %.2f", r)
	}
	if _, capped := remaining("nous", spent); capped {
		t.Error("nous should be uncapped")
	}
	if hasAllowance("go", spent) {
		t.Error("go should report no allowance")
	}
	if !hasAllowance("ollama", spent) {
		t.Error("ollama should report allowance")
	}
}
