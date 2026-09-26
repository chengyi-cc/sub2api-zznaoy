package basispoints

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// Codex documents exec's nested tools using separate TypeScript declarations.
// Only that explicit declaration grants a name; examples, history and arbitrary
// prose must never turn into callable aliases.
var execFunctionDeclaration = regexp.MustCompile("(?m)^[\\t ]*declare[\\t ]+const[\\t ]+tools[\\t ]*:[\\t ]*\\{[\\t \\r\\n]*([A-Za-z_][A-Za-z0-9_]*)[\\t ]*\\(")

func declaredExecFunctions(entry object) map[string]bool {
	name := text(entry["name"])
	if text(entry["type"]) != "custom" || (name != "exec" && name != "functions.exec") {
		return nil
	}
	description := text(entry["description"])
	if len(description) > maxEnvelopeBytes {
		return nil
	}
	var names map[string]bool
	for _, match := range execFunctionDeclaration.FindAllStringSubmatch(description, -1) {
		name := match[1]
		if name == "constructor" || name == "prototype" || name == "__proto__" {
			continue
		}
		if names == nil {
			names = make(map[string]bool)
		}
		names[name] = true
	}
	return names
}

// recoverExecCall repairs one skipped orchestration layer. It returns a custom
// call to the client's existing exec tool; the gateway executes nothing. Exact
// catalog calls are resolved first by the caller. Unknown or ambiguous nested
// names are never guessed, and raw code is never interpreted as a function call.
func (b *Bridge) recoverExecCall(native, envelope object) (object, bool, error) {
	if text(native["type"]) != "function_call" {
		return nil, false, nil
	}
	name, err := envelopeName(envelope)
	if err != nil {
		return nil, false, err
	}
	name = strings.TrimPrefix(name, "functions.")
	var host tool
	found := false
	for _, info := range b.tools {
		if !info.ExecFunctions[name] {
			continue
		}
		if found {
			return nil, true, fmt.Errorf("basispoints nested tool has ambiguous exec hosts")
		}
		host, found = info, true
	}
	if !found {
		return nil, false, nil
	}
	if _, exists := envelope["input"]; exists {
		return nil, true, fmt.Errorf("basispoints nested function requires arguments, not raw input")
	}
	args, err := envelopeArguments(envelope)
	if err != nil {
		return nil, true, err
	}
	if raw, ok := args.(string); ok {
		if len(raw) > maxEnvelopeBytes || decode([]byte(raw), &args) != nil {
			return nil, true, fmt.Errorf("basispoints nested function arguments are invalid JSON")
		}
	}
	if obj, ok := args.(object); !ok || obj == nil {
		return nil, true, fmt.Errorf("basispoints nested function arguments must be an object")
	}
	raw, err := json.Marshal(args)
	if err != nil || len(raw) > maxEnvelopeBytes {
		return nil, true, fmt.Errorf("basispoints nested function arguments exceed the transport contract")
	}
	// JSON.parse preserves keys such as __proto__ as data. Two JSON encodings
	// prevent quotes, newlines and executable-looking argument text becoming JS.
	literal, _ := json.Marshal(string(raw))
	callee, _ := json.Marshal(name)
	input := "text(await tools[" + string(callee) + "](JSON.parse(" + string(literal) + ")));"
	if len(input) > maxEnvelopeBytes {
		return nil, true, fmt.Errorf("basispoints nested function relay exceeds the size limit")
	}
	call, err := b.finishClientToolCall(native, host, object{"input": input}, true)
	return call, true, err
}
