package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// UsageStore keeps token counts per key per UTC day and forgets anything older
// than the retention window.
//
// The ledger answers "what did the providers cost this month"; this answers
// "which client spent it", which is a different question with a different
// lifetime — a month is the unit for allowances, but a key's history is only
// interesting for as long as someone might still reconcile it. Two months is
// the window: anything older is deleted rather than accumulated forever.
type UsageStore struct {
	mu            sync.Mutex
	path          string
	retentionDays int
	data          map[string]map[string]*KeyUsage // day "2006-01-02" -> label -> counters
}

// defaultUsageRetentionDays is two months, rounded to whole days: a month is
// the unit of an allowance, two of them is the unit of a dispute.
const defaultUsageRetentionDays = 60

// KeyUsage is one key's token counters for one day.
type KeyUsage struct {
	Prompt     int64 `json:"prompt"`
	CacheRead  int64 `json:"cache_read"`
	Completion int64 `json:"completion"`
	Requests   int64 `json:"requests"`
}

// Total is what a user means by "how many tokens did this key use": everything
// the provider charged for, cache reads included.
func (k KeyUsage) Total() int64 { return k.Prompt + k.CacheRead + k.Completion }

func (k *KeyUsage) add(o KeyUsage) {
	k.Prompt += o.Prompt
	k.CacheRead += o.CacheRead
	k.Completion += o.Completion
	k.Requests += o.Requests
}

func dayOf(t time.Time) string { return t.UTC().Format("2006-01-02") }

// NewUsageStore loads the store and immediately applies the retention window,
// so a file that sat unused for months is trimmed on the first start rather
// than on the first request.
func NewUsageStore(path string, retentionDays int) *UsageStore {
	if retentionDays <= 0 {
		retentionDays = defaultUsageRetentionDays
	}
	s := &UsageStore{
		path:          path,
		retentionDays: retentionDays,
		data:          map[string]map[string]*KeyUsage{},
	}
	if path != "" {
		if b, err := os.ReadFile(path); err == nil {
			_ = json.Unmarshal(b, &s.data)
		}
	}
	if s.data == nil {
		s.data = map[string]map[string]*KeyUsage{}
	}
	s.mu.Lock()
	if s.pruneLocked(time.Now().UTC()) > 0 {
		s.saveLocked()
	}
	s.mu.Unlock()
	return s
}

// Add records one completed request. A request that reported no usage still
// counts as a request, so a key that is failing shows up as traffic without
// tokens instead of vanishing.
func (s *UsageStore) Add(t time.Time, label string, u KeyUsage) {
	if s == nil {
		return
	}
	if label == "" {
		label = "anonymous"
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	day := dayOf(t)
	if s.data[day] == nil {
		s.data[day] = map[string]*KeyUsage{}
	}
	if s.data[day][label] == nil {
		s.data[day][label] = &KeyUsage{}
	}
	s.data[day][label].add(u)
	if s.pruneLocked(t) > 0 {
		// Pruning happens on writes so no timer is needed for it.
	}
	s.saveLocked()
}

// pruneLocked drops whole days that have fallen out of the window. Days are
// the unit of retention because a partially-kept day would understate a key.
func (s *UsageStore) pruneLocked(now time.Time) int {
	cutoff := now.UTC().AddDate(0, 0, -s.retentionDays).Format("2006-01-02")
	removed := 0
	for day := range s.data {
		if day < cutoff {
			delete(s.data, day)
			removed++
		}
	}
	return removed
}

// snapshot is what /v1/usage returns per key.
type snapshot struct {
	Label string   `json:"label"`
	Limit int64    `json:"monthly_token_limit"`
	Today KeyUsage `json:"today"`
	Month KeyUsage `json:"month"`
	Days  int      `json:"days_kept"`
}

// Snapshot sums the stored days into today's and this month's totals for every
// label seen inside the retention window.
func (s *UsageStore) Snapshot(now time.Time) []snapshot {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	today := dayOf(now)
	month := monthOf(now)
	acc := map[string]*snapshot{}
	for day, byLabel := range s.data {
		for label, u := range byLabel {
			row := acc[label]
			if row == nil {
				row = &snapshot{Label: label}
				acc[label] = row
			}
			row.Days++
			if strings.HasPrefix(day, month) {
				row.Month.add(*u)
			}
			if day == today {
				row.Today.add(*u)
			}
		}
	}
	for _, k := range keys {
		if _, ok := acc[k.Label]; !ok && k.MonthlyTokenLimit > 0 {
			acc[k.Label] = &snapshot{Label: k.Label}
		}
	}
	out := make([]snapshot, 0, len(acc))
	for label, row := range acc {
		for _, k := range keys {
			if k.Label == label {
				row.Limit = k.MonthlyTokenLimit
			}
		}
		// A day can hold several labels; Days is the label's own count, which
		// the loop above already accumulates per row.
		out = append(out, *row)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Label < out[j].Label })
	return out
}

func (s *UsageStore) saveLocked() {
	if s.path == "" {
		return
	}
	b, err := json.MarshalIndent(s.data, "", "  ")
	if err != nil {
		return
	}
	if dir := filepath.Dir(s.path); dir != "" {
		_ = os.MkdirAll(dir, 0o700)
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return
	}
	_ = os.Rename(tmp, s.path)
}
