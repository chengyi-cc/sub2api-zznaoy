package basispoints

import "strings"

// All transports share this resolver. Prefer exact declarations, then one host
// display prefix, then an unambiguous MCP display/runtime spelling. Never match
// arbitrary namespace suffixes or promote an undeclared name.
func resolveCatalogToolName(catalog map[string]tool, name string) (string, tool, bool) {
	if info, ok := catalog[name]; ok {
		return name, info, true
	}
	trimmed := strings.TrimPrefix(name, "functions.")
	if trimmed != name {
		if info, ok := catalog[trimmed]; ok {
			return trimmed, info, true
		}
	} else if info, ok := catalog["functions."+name]; ok {
		// functions is the host's default tool namespace, not an arbitrary
		// suffix alias. Exact bare declarations above retain precedence.
		return "functions." + name, info, true
	}
	method, mcp := execMCPMethod(trimmed)
	if !mcp {
		return "", tool{}, false
	}
	var key string
	var match tool
	for candidate, info := range catalog {
		normalized, ok := execMCPMethod(strings.TrimPrefix(candidate, "functions."))
		if !ok || normalized != method {
			continue
		}
		if key != "" {
			return "", tool{}, false
		}
		key, match = candidate, info
	}
	return key, match, key != ""
}
