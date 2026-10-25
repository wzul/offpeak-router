package main

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config is an optional JSON file that replaces the built-in provider table.
// Everything in it is optional: absent fields keep their compiled default, so
// a partial file is a valid override.
//
// Example:
//
//	{
//	  "listen": "127.0.0.1:8788",
//	  "ledger": "/var/lib/offpeak-router/spend.json",
//	  "fallback_order": ["cheap", "backup", "metered"],
//	  "providers": {
//	    "cheap": {
//	      "base_url": "https://opencode.ai/inference/go/openai/v1",
//	      "model": "deepseek-v4.1-flash",
//	      "api_key_env": "OPENCODE_API_KEY",
//	      "monthly_allowance_usd": 80,
//	      "rates": {"in": 0.15, "cache": 0.003, "out": 0.60, "peak_mult": 2.0},
//	      "peak_windows_utc": [[1, 4], [6, 10]]
//	    },
//	    "metered": {
//	      "base_url": "https://inference-api.nousresearch.com/v1",
//	      "model": "deepseek/deepseek-v4.1-flash-1017",
//	      "api_key_env": "NOUS_API_KEY",
//	      "rates": {"in": 0.02, "cache": 0.015, "out": 1.00, "peak_mult": 1.0}
//	    }
//	  }
//	}
//
// A provider with no monthly_allowance_usd is metered from the first token.
// A provider with no peak_windows_utc never peaks.
type Config struct {
	Listen        string                    `json:"listen"`
	Ledger        string                    `json:"ledger"`
	FallbackOrder []string                  `json:"fallback_order"`
	Providers     map[string]ProviderConfig `json:"providers"`
}

type ProviderConfig struct {
	Name         string   `json:"name"`
	BaseURL      string   `json:"base_url"`
	Model        string   `json:"model"`
	APIKeyEnv    string   `json:"api_key_env"`
	MonthlyAllow *float64 `json:"monthly_allowance_usd"`
	// HardCap means the allowance cannot be exceeded: once spent, the provider
	// is dropped. Omit it for a subscription you can top up past the allowance.
	HardCap        bool     `json:"hard_cap"`
	Rates          *Rates   `json:"rates"`
	PeakWindowsUTC [][2]int `json:"peak_windows_utc"`
}

func loadConfig(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("parse config %s: %w", path, err)
	}
	return &c, nil
}

// applyConfig replaces the built-in tables with whatever the file specifies.
// A provider entry replaces that provider wholesale, so a rates block that
// omits peak_mult silently becomes 0 rather than 2.0 — validate loudly.
func applyConfig(c *Config) error {
	if c.Listen != "" {
		cfgListen = c.Listen
	}
	if c.Ledger != "" {
		cfgLedger = c.Ledger
	}
	if len(c.FallbackOrder) > 0 {
		failoverChain = c.FallbackOrder
	}
	if len(c.Providers) == 0 {
		return nil
	}

	newProvs := map[string]*Provider{}
	newWindows := map[string][]Window{}
	newAllowance := map[string]float64{}
	newRates := map[string]Rates{}
	newHardCap := map[string]bool{}

	for name, pc := range c.Providers {
		if pc.BaseURL == "" || pc.Model == "" || pc.APIKeyEnv == "" {
			return fmt.Errorf("provider %q needs base_url, model and api_key_env", name)
		}
		if pc.Rates == nil {
			return fmt.Errorf("provider %q needs a rates block", name)
		}
		if pc.Rates.PeakMult == 0 {
			return fmt.Errorf("provider %q has peak_mult 0 — omit the field for a "+
				"provider that never peaks, or set 1.0 for one that does not double", name)
		}
		for _, w := range pc.PeakWindowsUTC {
			if w[0] < 0 || w[0] > 23 || w[1] < 1 || w[1] > 24 || w[1] <= w[0] {
				return fmt.Errorf("provider %q has an invalid peak window [%d,%d)", name, w[0], w[1])
			}
		}

		newProvs[name] = &Provider{Base: pc.BaseURL, Model: pc.Model, KeyEnv: pc.APIKeyEnv}
		wins := make([]Window, 0, len(pc.PeakWindowsUTC))
		for _, w := range pc.PeakWindowsUTC {
			wins = append(wins, Window{w[0], w[1]})
		}
		newWindows[name] = wins
		if pc.MonthlyAllow != nil {
			newAllowance[name] = *pc.MonthlyAllow
		}
		if pc.HardCap {
			newHardCap[name] = true
		}
		newRates[name] = *pc.Rates
	}

	provs = newProvs
	peakWindows = newWindows
	allowance = newAllowance
	rates = newRates
	hardCap = newHardCap
	return nil
}
