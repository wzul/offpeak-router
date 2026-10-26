package main

import (
	"sort"
	"time"
)

// Window is a half-open UTC hour range [Start, End).
type Window struct{ Start, End int }

// peakWindows is Monday-Friday UTC only; weekends are fully off-peak.
// The Go and Ollama windows are disjoint, so at least one provider is always
// off-peak on a weekday.
var peakWindows = map[string][]Window{
	"go":     {{1, 4}, {6, 10}},
	"ollama": {{12, 18}},
	"nous":   {},
}

// allowance is the monthly included usage in USD. A missing key means the
// provider is metered from the first token.
var allowance = map[string]float64{
	"go":     80.0, // $10 plan -> $80 of usage
	"ollama": 60.0, // $20 Pro  -> $60 credit
}

type Rates struct {
	In       float64 `json:"in"`
	Cache    float64 `json:"cache"`
	Out      float64 `json:"out"`
	PeakMult float64 `json:"peak_mult"`
}

var rates = map[string]Rates{
	"go":     {0.15, 0.003, 0.60, 2.0},
	"ollama": {0.15, 0.003, 0.60, 2.0},
	"nous":   {0.02, 0.015, 1.00, 1.0},
}

// hardCap marks a provider whose monthly allowance cannot be exceeded: once it
// is used up the provider is dropped from the chain entirely. Providers without
// this flag keep serving past their allowance at the plain marginal rate (a
// top-up), which is often still cheaper than a third provider.
var hardCap = map[string]bool{"go": true}

var failoverChain = []string{"go", "ollama", "nous"}

func isPeak(provider string, now time.Time) bool {
	windows, ok := peakWindows[provider]
	if !ok || len(windows) == 0 {
		return false
	}
	if wd := now.Weekday(); wd == time.Saturday || wd == time.Sunday {
		return false
	}
	h := now.Hour()
	for _, w := range windows {
		if h >= w.Start && h < w.End {
			return true
		}
	}
	return false
}

// ratesFor returns the rate card to use for a provider serving a particular
// canonical model. A per-model override on that provider wins, because one
// provider can host several models at very different prices.
func ratesFor(provider, model string) (Rates, bool) {
	if model != "" {
		if p, ok := provs[provider]; ok {
			if r, ok := p.Models[model]; ok && r.Rates != nil {
				return *r.Rates, true
			}
		}
	}
	r, ok := rates[provider]
	return r, ok
}

func requestCost(provider, model string, inTok, cacheTok, outTok float64, peak bool) float64 {
	r, ok := ratesFor(provider, model)
	if !ok {
		return 0
	}
	m := 1.0
	if peak {
		m = r.PeakMult
	}
	return m * (inTok*r.In + cacheTok*r.Cache + outTok*r.Out) / 1e6
}

func remaining(provider string, spent map[string]float64) (float64, bool) {
	cap, capped := allowance[provider]
	if !capped {
		return 0, false // uncapped
	}
	r := cap - spent[provider]
	if r < 0 {
		r = 0
	}
	return r, true
}

func hasAllowance(provider string, spent map[string]float64) bool {
	r, capped := remaining(provider, spent)
	return !capped || r > 0
}

// servesModel reports whether a provider can serve the canonical model. A
// provider with no explicit model table serves exactly one model, its Model.
func servesModel(provider, model string) bool {
	p, ok := provs[provider]
	if !ok {
		return false
	}
	if len(p.Models) == 0 {
		return p.Model == model
	}
	_, ok = p.Models[model]
	return ok
}

// upstreamModel maps a canonical model to the id that provider expects. This is
// why the model table exists: the same model is spelled differently by each
// provider (deepseek-v4.1-flash vs deepseek-v4.1-flash:1017), and a client
// asking for one name must not have to track the others.
func upstreamModel(provider, canonical string) string {
	p, ok := provs[provider]
	if !ok {
		return canonical
	}
	if len(p.Models) == 0 {
		return p.Model
	}
	if r, ok := p.Models[canonical]; ok && r.ID != "" {
		return r.ID
	}
	return canonical
}

// candidateOrder returns provider names best-first.
//
// model is the canonical model the request asked for (already resolved). An
// empty model means "do not filter", which the policy tests use to prove the
// ordering itself names no provider.
//
// Tiers, in order:
//
//  1. allowance remaining AND off-peak — the allowance covers ~2x more requests here
//  2. allowance remaining at peak       — a prepaid token still beats a metered one
//  3. metered: no allowance, top-up, or uncapped — ranked by current marginal cost
//
// A hard-capped provider whose allowance is spent is dropped entirely: it cannot
// serve at any price.
//
// Nothing here names a provider. It works for whatever set is loaded.
type cand struct {
	name string
	cost float64
}

func candidateOrder(now time.Time, spent map[string]float64, model string) []string {
	var offPeak, atPeak, metered []cand
	// usable records which providers may serve at any price. The failover chain
	// is appended last and must not resurrect one that was deliberately dropped.
	usable := make(map[string]bool, len(provs))

	for name := range provs {
		if model != "" && !servesModel(name, model) {
			continue
		}
		peak := isPeak(name, now)
		cost := requestCost(name, model, 6801, 170536, 880, peak)
		r, capped := remaining(name, spent)

		switch {
		case !capped:
			// No allowance at all: metered from the first token.
			metered = append(metered, cand{name, cost})
		case r > 0 && !peak:
			offPeak = append(offPeak, cand{name, cost})
		case r > 0:
			atPeak = append(atPeak, cand{name, cost})
		case hardCap[name]:
			// Allowance spent and no top-up possible: unusable, of any price.
			continue
		default:
			metered = append(metered, cand{name, cost})
		}
		usable[name] = true
	}

	sortByCost(offPeak)
	sortByCost(atPeak)
	sortByCost(metered)

	order := make([]string, 0, len(provs))
	for _, group := range [][]cand{offPeak, atPeak, metered} {
		for _, c := range group {
			order = append(order, c.name)
		}
	}

	// A provider marked First is tried before all of it: ahead of the peak group it
	// would otherwise land in, ahead of the price sort, and ahead of the failover
	// chain. It is still subject to `usable` (it must serve this model) and to the
	// circuit breaker, which is what makes "until it errors" work: one failure moves
	// the request on, repeated failures open the breaker and skip it entirely.
	first := make([]cand, 0, 1)
	for name := range provs {
		if !provs[name].First || !usable[name] {
			continue
		}
		first = append(first, cand{name, 0})
	}
	sort.SliceStable(first, func(i, j int) bool { return first[i].name < first[j].name })

	// Append the failover chain, ignoring names that are not loaded, and
	// de-duplicate so every provider appears exactly once.
	seen := make(map[string]bool, len(provs))
	out := make([]string, 0, len(provs))
	pinned := make([]string, 0, len(first))
	for _, c := range first {
		seen[c.name] = true
		pinned = append(pinned, c.name)
	}
	out = append(out, pinned...)
	for _, p := range append(order, failoverChain...) {
		if !usable[p] || seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, p)
	}
	return out
}

// chainRank is the provider's position in the configured failover order, used
// only to break ties between providers that cost exactly the same right now.
func chainRank(name string) int {
	for i, p := range failoverChain {
		if p == name {
			return i
		}
	}
	return len(failoverChain)
}

// sortByCost orders candidates by price, then by the configured failover
// preference, then by name.
//
// The tiebreak is not cosmetic: providers are collected by ranging over a map,
// so without a total order two providers priced identically would be picked in
// a different order on every request.
func sortByCost(cs []cand) {
	sort.SliceStable(cs, func(i, j int) bool {
		if cs[i].cost != cs[j].cost {
			return cs[i].cost < cs[j].cost
		}
		ri, rj := chainRank(cs[i].name), chainRank(cs[j].name)
		if ri != rj {
			return ri < rj
		}
		return cs[i].name < cs[j].name
	})
}
