package main

import (
	"encoding/json"
	"os"
	"sync"
	"time"
)

// Ledger keeps month-to-date spend per provider, in USD of usage. The
// allowances are monthly and do not roll over, so the key is the calendar
// month: a new month simply starts a fresh bucket.
//
// Backed by a small JSON file written atomically — three counters do not
// justify a SQLite dependency, and a pure-Go SQLite driver would add ~10MB
// to the binary for no benefit here.
type Ledger struct {
	mu   sync.Mutex
	path string
	data map[string]map[string]float64 // "2006-01" -> provider -> usd
}

func NewLedger(path string) *Ledger {
	l := &Ledger{path: path, data: map[string]map[string]float64{}}
	if b, err := os.ReadFile(path); err == nil {
		_ = json.Unmarshal(b, &l.data)
	}
	if l.data == nil {
		l.data = map[string]map[string]float64{}
	}
	return l
}

func monthOf(t time.Time) string { return t.UTC().Format("2006-01") }

// Spent returns a copy so callers never race with a concurrent Add.
func (l *Ledger) Spent(t time.Time) map[string]float64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	// Only what has actually been spent. A missing provider is a map zero, so
	// the ledger stays independent of whichever providers are loaded.
	out := make(map[string]float64, len(l.data[monthOf(t)]))
	for k, v := range l.data[monthOf(t)] {
		out[k] = v
	}
	return out
}

func (l *Ledger) Add(t time.Time, provider string, usd float64) {
	if usd <= 0 {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	mk := monthOf(t)
	if l.data[mk] == nil {
		l.data[mk] = map[string]float64{}
	}
	l.data[mk][provider] += usd
	l.saveLocked()
}

func (l *Ledger) saveLocked() {
	b, err := json.MarshalIndent(l.data, "", "  ")
	if err != nil {
		return
	}
	tmp := l.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return
	}
	_ = os.Rename(tmp, l.path)
}
