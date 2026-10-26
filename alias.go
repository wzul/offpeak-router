package main

// Claude model ids.
//
// A Claude Code session sends model names this router does not serve: the top-level
// model it was configured with, and - more importantly - the small model it uses
// internally for summaries, titles and fetch digests. Those requests arrive before any
// search, so an unknown-model 400 makes the whole session fail for a reason that has
// nothing to do with the answer.
//
// Every Claude id therefore lands on the canonical chat model. The mapping is a prefix
// rule, not a table: new dated ids (claude-sonnet-4-5-20250929, claude-opus-5-5, ...)
// keep working without a code change, and no request is silently rerouted to something
// expensive. Ids that are not Claude-shaped still fail loudly, so a typo stays a typo.

import "strings"

// isClaudeModelID reports whether a requested id names a Claude model.
func isClaudeModelID(requested string) bool {
	id := strings.ToLower(strings.TrimSpace(requested))
	if id == "" {
		return false
	}
	if id == "claude" {
		return true
	}
	// "claude-...", "claude...." and "claude_..." all appear in the wild: the API ids
	// use dashes, some SDK wrappers use dots, and provider proxies use underscores.
	for _, sep := range []string{"-", ".", "_", " ", "/"} {
		if strings.HasPrefix(id, "claude"+sep) {
			return true
		}
	}
	return false
}

// claudeAliasTarget is the model a Claude id resolves to, or "" when this router does
// not serve a chat model to alias onto.
func claudeAliasTarget() string {
	if _, ok := resolveModel(canonicalModel); !ok {
		return ""
	}
	if api := modelAPI(canonicalModel); api != APIChat {
		return ""
	}
	return canonicalModel
}
