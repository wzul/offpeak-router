package main

// Model capabilities, so discovery can answer "can this thing see images?"
//
// It could not before: /v1/models reported an id, a provider list and the API
// family, and nothing about modality. An agent looking for a vision model found
// no evidence of one and reasonably concluded the router had none - while the
// model behind it answered "Red" for a red PNG on two different upstreams.
//
// A capability is listed only after a live request proved it, and the providers
// it was proved against are published alongside: an unverified upstream is not
// the same as a capable one, and a client that routes to it deserves to know.

// modelCapabilities holds the verified capability set per canonical model id.
var modelCapabilities = map[string][]string{
	"deepseek-v4.1-flash": {"completion", "vision", "thinking"},
	// gemma4:31b answered "Red" for a red PNG on ollama; go does not serve it at
	// all, and no other upstream was verified.
	"gemma4": {"completion", "vision"},
	// Clef answers one decision schema; it is not a chat or a vision model.
	"clef": {"completion"},
}

// capabilityProviders records which upstream each capability was verified on.
// A provider absent from a capability's list is unverified, not proven unable.
var capabilityProviders = map[string]map[string][]string{
	"deepseek-v4.1-flash": {
		"vision":   {"go", "ollama"},
		"thinking": {"go", "ollama"},
	},
	"gemma4": {
		"vision": {"ollama"},
	},
}

// inputModalities derives the modality list from the capabilities, since clients
// that follow the OpenAI style ask for that field rather than named capabilities.
func inputModalities(model string) []string {
	mods := []string{"text"}
	for _, c := range capabilitiesFor(model) {
		if c == "vision" {
			mods = append(mods, "image")
			break
		}
	}
	return mods
}

func capabilitiesFor(model string) []string {
	if caps, ok := modelCapabilities[model]; ok {
		return caps
	}
	return []string{"completion"}
}

// capabilityProvidersFor narrows the per-capability provider map to the model.
func capabilityProvidersFor(model string) map[string][]string {
	if m, ok := capabilityProviders[model]; ok {
		return m
	}
	return map[string][]string{}
}

// hasVision reports whether any verified upstream can take images for this model.
func hasVision(model string) bool {
	for _, c := range capabilitiesFor(model) {
		if c == "vision" {
			return true
		}
	}
	return false
}
