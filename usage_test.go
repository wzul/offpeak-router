package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestUsageStoreTracksPerKeyPerDay(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "usage.json")
	s := NewUsageStore(path, 60)

	now := time.Date(2026, 11, 9, 10, 0, 0, 0, time.UTC)
	s.Add(now, "fleet", KeyUsage{Prompt: 100, CacheRead: 20, Completion: 30, Requests: 1})
	s.Add(now, "fleet", KeyUsage{Prompt: 50, Completion: 10, Requests: 1})
	s.Add(now, "editor", KeyUsage{Prompt: 7, Completion: 3, Requests: 1})

	rows := s.Snapshot(now)
	if len(rows) != 2 {
		t.Fatalf("want 2 labels, got %d: %+v", len(rows), rows)
	}
	if rows[0].Label != "editor" || rows[1].Label != "fleet" {
		t.Fatalf("labels not sorted: %+v", rows)
	}
	fleet := rows[1]
	if fleet.Today.Total() != 210 {
		t.Fatalf("fleet total = %d; want 210", fleet.Today.Total())
	}
	if fleet.Today.Requests != 2 {
		t.Fatalf("fleet requests = %d; want 2", fleet.Today.Requests)
	}
	if fleet.Month.Total() != 210 {
		t.Fatalf("fleet month = %d; want 210", fleet.Month.Total())
	}
}

// Two months is the promise: a day older than the window is deleted, and the
// month total must not include days that fall outside the current month.
func TestUsageStorePrunesBeyondRetention(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.json")
	s := NewUsageStore(path, 60)

	now := time.Date(2026, 11, 9, 10, 0, 0, 0, time.UTC)
	s.Add(now.AddDate(0, 0, -61), "fleet", KeyUsage{Prompt: 1000, Requests: 1}) // too old
	s.Add(now.AddDate(0, 0, -59), "fleet", KeyUsage{Prompt: 500, Requests: 1})  // kept
	s.Add(now.AddDate(0, 0, -10), "fleet", KeyUsage{Prompt: 40, Requests: 1})   // last month, kept
	s.Add(now, "fleet", KeyUsage{Prompt: 5, Requests: 1})                       // today

	rows := s.Snapshot(now)
	if len(rows) != 1 {
		t.Fatalf("want 1 label, got %+v", rows)
	}
	if got := rows[0].Today.Total(); got != 5 {
		t.Fatalf("today = %d; want 5", got)
	}
	if got := rows[0].Month.Total(); got != 5 {
		t.Fatalf("month = %d; want 5 (only today is in November)", got)
	}
	if rows[0].Days != 3 {
		t.Fatalf("days kept = %d; want 3", rows[0].Days)
	}
}

// A restart must not lose the counters, and must apply retention on load.
func TestUsageStorePersistsAndPrunesOnLoad(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.json")
	now := time.Now().UTC()

	s := NewUsageStore(path, 60)
	s.Add(now, "fleet", KeyUsage{Prompt: 1234, Requests: 1})
	s.Add(now.AddDate(0, 0, -200), "old", KeyUsage{Prompt: 99, Requests: 1})
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("usage file not written: %v", err)
	}

	reloaded := NewUsageStore(path, 60)
	rows := reloaded.Snapshot(now)
	if len(rows) != 1 || rows[0].Label != "fleet" {
		t.Fatalf("reload lost data or kept a stale day: %+v", rows)
	}
	if rows[0].Today.Total() != 1234 {
		t.Fatalf("reloaded total = %d; want 1234", rows[0].Today.Total())
	}
}

// A key with no traffic yet still reports its limit, so /v1/usage can show an
// allowance before it is first spent.
func TestSnapshotIncludesUnusedKeyWithLimit(t *testing.T) {
	defer func(prev []APIKey) { keys = prev }(keys)
	keys = []APIKey{{Label: "editor", Token: "t", MonthlyTokenLimit: 1000}}

	s := NewUsageStore(filepath.Join(t.TempDir(), "usage.json"), 60)
	rows := s.Snapshot(time.Now().UTC())
	if len(rows) != 1 || rows[0].Label != "editor" || rows[0].Limit != 1000 {
		t.Fatalf("want editor with limit 1000, got %+v", rows)
	}
}
