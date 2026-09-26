package basispoints

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strconv"
	"strings"
)

const maxToolRepairs = 2

// ToolRepairFunc continues the same BPS conversation after a rejected native
// transport call. The entire tool batch is still unexecuted at this boundary.
type ToolRepairFunc func(context.Context, map[string]any, error) (map[string]any, error)

func (b *Bridge) validateToolResponse(response object) error {
	check := *b
	check.replay = nil
	output, _ := response["output"].([]any)
	ids, calls := make(map[string]bool), make(map[string]bool)
	count := 0
	for _, raw := range output {
		item, _ := raw.(object)
		if !isTool(item) {
			continue
		}
		count++
		if count > 1024 {
			return fmt.Errorf("basispoints returned too many tool calls")
		}
		// Translation may normalize nested maps; isolate the dry run completely.
		item = copyToolObject(item)
		translated, err := check.translateCall(item)
		if err != nil {
			return err
		}
		id, call := text(translated["id"]), text(translated["call_id"])
		if ids[id] || calls[call] {
			return fmt.Errorf("basispoints response contains duplicate tool identities")
		}
		ids[id], calls[call] = true, true
	}
	if !b.parallelTools && count > 1 {
		return fmt.Errorf("basispoints returned parallel tool calls when the client disabled them; no tools were dispatched")
	}
	return nil
}

func copyToolObject(value object) object {
	raw, _ := json.Marshal(value)
	var out object
	_ = decode(raw, &out)
	return out
}

func nativePlanCall(item object) bool {
	name := text(item["name"])
	ns := text(item["namespace"])
	return text(item["type"]) == "function_call" && (ns == "" || ns == "functions") && (name == "update_plan" || name == "functions.update_plan")
}

// Only complete BPS wrapper batches can be returned as native tool errors.
// Never infer a target for arbitrary code, invent a call ID, or retry I/O.
func repairableTools(response object) ([]object, bool) {
	if status := text(response["status"]); status != "" && status != "completed" {
		return nil, false
	}
	output, _ := response["output"].([]any)
	var items []object
	ids := make(map[string]bool)
	for _, raw := range output {
		item, _ := raw.(object)
		if !isTool(item) {
			continue
		}
		name, call := text(item["name"]), text(item["call_id"])
		if text(item["type"]) != "function_call" || (name != "run_officejs" && name != "functions.run_officejs" && !nativePlanCall(item)) || call == "" || ids[call] {
			return nil, false
		}
		args, ok := item["arguments"].(object)
		if !ok && decode([]byte(text(item["arguments"])), &args) != nil {
			return nil, false
		}
		if args == nil {
			return nil, false
		}
		ids[call] = true
		items = append(items, item)
	}
	return items, len(items) > 0 && len(items) <= 1024
}

func (b *Bridge) translateCompleted(ctx context.Context, response object, repair ToolRepairFunc) error {
	validation := b.validateToolResponse(response)
	if validation == nil {
		return b.translateResponse(response)
	}
	if b.observeToolFailure != nil {
		b.observeToolFailure(response, validation)
	}
	original, eligible := repairableTools(response)
	if repair == nil || !eligible || b.structured != nil {
		return validation
	}
	for _, item := range original {
		if target := transportTarget(transportArguments(item)); target != "" {
			if _, allowed := b.resolveCatalogTool(target); !allowed {
				return validation
			}
		}
	}
	// A native plan may be bookkeeping for a capability absent from this client.
	// Let the model return a textual plan, but never substitute a different action.
	planOnly := len(original) == 1 && nativePlanCall(original[0])
	// When parallel calls are forbidden, only the unchanged first operation may
	// be issued. Remaining operations stay unexecuted until a later model turn.
	serial := !b.parallelTools && len(original) > 1
	if serial {
		seen := false
		output, _ := response["output"].([]any)
		for _, raw := range output {
			item, _ := raw.(object)
			if isTool(item) {
				seen = true
			} else if seen {
				return validation
			}
		}
	}
	failed := response
	usage := addRepairUsage(nil, response["usage"])
	for attempt := 0; attempt < maxToolRepairs; attempt++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		corrected, err := repair(ctx, failed, validation)
		usage = addRepairUsage(usage, corrected["usage"])
		if usage != nil {
			response["usage"] = usage
		}
		if err != nil {
			return fmt.Errorf("basispoints tool transport correction failed: %w", err)
		}
		items, ok := repairableTools(corrected)
		if planOnly && !ok && hasOnlyPlanText(corrected) {
			output, _ := response["output"].([]any)
			for i, raw := range output {
				item, _ := raw.(object)
				if isTool(item) {
					// The added textual item must be emitted explicitly by the stream.
					msgs, _ := corrected["output"].([]any)
					for _, msg := range msgs {
						m, _ := msg.(object)
						if text(m["type"]) == "message" {
							output[i] = m
							break
						}
					}
				}
			}
			response["output"] = output
			return nil
		}
		want := len(original)
		if serial {
			want = 1
		}
		if !ok || len(items) != want {
			return b.observeCorrectionFailure(corrected, fmt.Errorf("basispoints tool transport correction changed the tool batch; no tool was executed"))
		}
		validation = b.validateToolResponse(corrected)
		if validation != nil && b.observeToolFailure != nil {
			b.observeToolFailure(corrected, validation)
		}
		if validation == nil {
			compare := original
			if serial {
				compare = original[:1]
			}
			items, err = b.restoreRawToolPayloads(compare, items)
			if err != nil {
				return b.observeCorrectionFailure(corrected, err)
			}
			if !b.preservesToolOperations(compare, items) {
				return b.observeCorrectionFailure(corrected, fmt.Errorf("basispoints tool transport correction changed an operation; no tool was executed"))
			}
			// Text has already streamed. Replace only its withheld tool slots,
			// keeping one downstream response identity and stable output indexes.
			output, _ := response["output"].([]any)
			next := 0
			kept := make([]any, 0, len(output))
			for i, raw := range output {
				item, _ := raw.(object)
				if isTool(item) {
					if next >= len(items) {
						continue
					}
					output[i] = items[next]
					next++
				}
				kept = append(kept, output[i])
			}
			response["output"] = kept
			return b.translateResponse(response)
		}
		failed = corrected
	}
	return fmt.Errorf("basispoints tool transport remains invalid after %d corrections; no tool was executed: %w", maxToolRepairs, validation)
}

// Keep the rejected correction as well as the initial invalid call. Otherwise
// the archive cannot explain why an individually valid correction was refused.
func (b *Bridge) observeCorrectionFailure(corrected object, err error) error {
	if b.observeToolFailure != nil {
		b.observeToolFailure(corrected, err)
	}
	return err
}

func hasOnlyPlanText(response object) bool {
	if text(response["status"]) != "completed" {
		return false
	}
	output, _ := response["output"].([]any)
	count := 0
	for _, raw := range output {
		item, _ := raw.(object)
		switch text(item["type"]) {
		case "reasoning":
		case "message":
			if text(item["role"]) != "assistant" {
				return false
			}
			content, _ := item["content"].([]any)
			for _, part := range content {
				p, _ := part.(object)
				if text(p["type"]) != "output_text" || strings.TrimSpace(text(p["text"])) == "" {
					return false
				}
			}
			if len(content) == 0 {
				return false
			}
			count++
		default:
			return false
		}
	}
	return count == 1
}

// Valid calls in a mixed batch must retain their exact client operation. For
// unframed raw text, require byte-for-byte preservation in the corrected payload.
// This checks content, without interpreting code or choosing its target.
func (b *Bridge) preservesToolOperations(original, corrected []object) bool {
	check := *b
	check.replay = nil
	for i, native := range original {
		before, err := check.translateCall(copyToolObject(native))
		after, afterErr := check.translateCall(copyToolObject(corrected[i]))
		if afterErr != nil {
			return false
		}
		if nativePlanCall(native) && err != nil {
			if text(after["type"]) != "function_call" || text(after["name"]) != "update_plan" {
				return false
			}
			var afterArgs object
			if decode([]byte(text(after["arguments"])), &afterArgs) != nil {
				return false
			}
			a, aok := planOperationSignature(transportArguments(native))
			z, zok := planOperationSignature(afterArgs)
			if !aok || !zok || !reflect.DeepEqual(a, z) {
				return false
			}
			continue
		}
		if err == nil {
			// A call ID changes between model turns; it is not part of the operation.
			before["call_id"], after["call_id"] = "compare", "compare"
			if historyCallFingerprint(before) != historyCallFingerprint(after) {
				return false
			}
			continue
		}
		args := transportArguments(native)
		if target := transportTarget(args); target != "" {
			info, allowed := b.resolveCatalogTool(target)
			if !allowed || after["name"] != info.Name || text(after["namespace"]) != info.Namespace {
				return false
			}
		}
		code, isText := args["code"].(string)
		if !isText {
			return false
		}
		if envelope, envelopeErr := decodeTransportEnvelope(code); envelopeErr == nil {
			if name, nameErr := envelopeName(envelope); nameErr == nil && name != "" {
				// A decodable envelope cannot authorize changed arguments merely
				// because its target name remained the same.
				expected, valid := recoverEnvelopeOperation(envelope, after)
				if !valid || !expected {
					return false
				}
				continue
			}
		}
		if text(after["type"]) == "custom_tool_call" {
			if after["input"] != code {
				return false
			}
		} else {
			var payload object
			if decode([]byte(text(after["arguments"])), &payload) != nil || payload["code"] != code {
				return false
			}
			// Metadata such as a working directory or target also belongs to the
			// operation. Repairing a missing marker cannot authorize new metadata.
			expected := object{}
			metadata := text(args["extended_summary"])
			if decode([]byte(metadata), &expected) != nil || expected == nil {
				if strings.HasPrefix(text(args["summary"]), functionCodeTransportPrefix) {
					return false
				}
				expected = object{}
			}
			if existing, exists := expected["code"]; exists && existing != code {
				return false
			}
			expected["code"] = code
			if !reflect.DeepEqual(expected, payload) {
				return false
			}
		}
	}
	return true
}

func planOperationSignature(args object) (object, bool) {
	steps, ok := args["plan"].([]any)
	if !ok || len(steps) == 0 {
		return nil, false
	}
	result := object{}
	normalized := make([]any, 0, len(steps))
	for _, raw := range steps {
		step, ok := raw.(object)
		if !ok {
			return nil, false
		}
		description, present, err := planTextAlias(step, "step", "description", "title")
		status := normalizeNativePlanStatus(text(step["status"]))
		if err != nil || !present || description == "" || status == "" {
			return nil, false
		}
		normalized = append(normalized, object{"step": description, "status": status})
	}
	result["plan"] = normalized
	explanation, present, err := planTextAlias(args, "explanation", "summary")
	if err != nil {
		return nil, false
	}
	if present {
		result["explanation"] = explanation
	}
	return result, true
}

func recoverEnvelopeOperation(envelope, after object) (bool, bool) {
	if text(after["type"]) == "custom_tool_call" {
		v, present := envelope["input"]
		if !present {
			v, present = envelope["args"]
		}
		return reflect.DeepEqual(v, after["input"]), present
	}
	var args any
	args, err := envelopeArguments(envelope)
	if err != nil {
		return false, false
	}
	if s, ok := args.(string); ok {
		if decode([]byte(s), &args) != nil {
			return false, false
		}
	}
	var actual any
	if decode([]byte(text(after["arguments"])), &actual) != nil {
		return false, false
	}
	return reflect.DeepEqual(args, actual), true
}

func transportArguments(native object) object {
	if args, ok := native["arguments"].(object); ok {
		return args
	}
	var args object
	_ = decode([]byte(text(native["arguments"])), &args)
	return args
}

func transportTarget(args object) string {
	for _, prefix := range []string{customTransportPrefix, functionCodeTransportPrefix} {
		if summary := text(args["summary"]); strings.HasPrefix(summary, prefix) {
			return strings.TrimPrefix(summary, prefix)
		}
	}
	if envelope, err := decodeTransportEnvelope(args["code"]); err == nil {
		name, _ := envelopeName(envelope)
		return name
	}
	return ""
}

// Sum billable usage across model continuations, including failed corrections.
func addRepairUsage(total object, value any) object {
	usage, _ := value.(object)
	if usage == nil {
		return total
	}
	result := make(object)
	for k, v := range total {
		result[k] = v
	}
	for k, v := range usage {
		if nested, ok := v.(object); ok {
			previous, _ := result[k].(object)
			result[k] = addRepairUsage(previous, nested)
			continue
		}
		var n int64
		var err error
		switch v := v.(type) {
		case json.Number:
			n, err = v.Int64()
		case int:
			n = int64(v)
		case int64:
			n = v
		default:
			continue
		}
		if err == nil && n >= 0 {
			previous, _ := result[k].(int64)
			result[k] = previous + n
		}
	}
	return result
}

// BuildToolRepairRequest extends an already prepared BPS request. It preserves
// the model, effort, scope and native history; no client tool is executed here.
func BuildToolRepairRequest(prepared []byte, failed map[string]any, validation error) ([]byte, error) {
	items, ok := repairableTools(failed)
	if !ok || validation == nil {
		return nil, fmt.Errorf("basispoints tool response is not eligible for correction")
	}
	var request object
	if decode(prepared, &request) != nil || request == nil {
		return nil, fmt.Errorf("invalid prepared Basispoints request")
	}
	input, ok := request["input"].([]any)
	if !ok {
		return nil, fmt.Errorf("basispoints correction requires expanded history")
	}
	output, _ := failed["output"].([]any)
	input = append(input, output...)
	feedback, _ := json.Marshal(object{"executed": false, "error": object{"code": "invalid_client_tool_transport", "message": validation.Error()}})
	for _, item := range items {
		call := text(item["call_id"])
		input = append(input, object{"type": "function_call_output", "id": "fc_" + fingerprint([]any{call, len(input)}), "call_id": call, "output": string(feedback)})
	}
	input = append(input, message("developer", fmt.Sprintf("The preceding tool batch failed transport validation before any client tool was executed. Correct only its transport formatting and return exactly %d run_officejs calls in the same order, preserving the intended operations and exact raw code. Use the existing client catalog: FUNCTION needs one JSON envelope with object arguments; CUSTOM needs summary=codex2api.custom/CATALOG_NAME and raw input in code; FUNCTION_CODE needs its declared marker and metadata JSON in extended_summary. Do not execute Office code, infer an undeclared target, add operations, or repeat commentary.", len(items))))
	if len(items) == 1 && nativePlanCall(items[0]) {
		input = append(input, message("developer", "The native update_plan tool is unavailable or incompatible with the client's declared plan schema. Use the exact declared client plan tool and schema if available. Otherwise return the intended plan as one short assistant text message; do not claim the plan tool ran, do not call any other tool, and do not change the user's task."))
	}
	if strings.Contains(validation.Error(), "parallel tool calls when the client disabled") {
		input = append(input, message("developer", "The client permits at most one tool call in this response. Override the preceding batch-count instruction: return only the FIRST original client operation, with identical target, arguments and raw code, through run_officejs. Do not execute or bundle the remaining operations; reconsider them after receiving the first real result on the next turn. Do not claim that any of this batch has executed."))
	}
	request["input"] = input
	metadata, _ := request["metadata"].(object)
	if metadata == nil {
		return nil, fmt.Errorf("basispoints correction requires request metadata")
	}
	iteration, err := strconv.Atoi(text(metadata["agent_iteration"]))
	if err != nil {
		return nil, fmt.Errorf("invalid Basispoints agent iteration")
	}
	metadata["agent_iteration"] = strconv.Itoa(iteration + 1)
	return json.Marshal(request)
}

// ReadToolRepairResponse withholds all intermediate events from the client.
// Only the terminal native response is authoritative, as in the primary stream.
func ReadToolRepairResponse(reader io.Reader) (map[string]any, error) {
	var response object
	var terminalError error
	pending := make(map[string]bool)
	err := readEvents(io.LimitReader(reader, 32<<20), func(event string, data []byte) error {
		if string(data) == "[DONE]" {
			return nil
		}
		var payload object
		if decode(data, &payload) != nil || payload == nil {
			return fmt.Errorf("invalid Basispoints correction SSE event")
		}
		kind := text(payload["type"])
		if kind == "" {
			kind = event
		}
		item, _ := payload["item"].(object)
		if kind == "response.output_item.done" && isTool(item) {
			if len(pending) >= 1024 {
				return fmt.Errorf("too many Basispoints correction tool items")
			}
			pending[text(item["call_id"])+"\x00"+text(item["id"])] = true
		}
		switch kind {
		case "response.completed", "response.failed", "response.incomplete", "error":
			response, _ = payload["response"].(object)
			if kind != "response.completed" || response == nil {
				terminalError = fmt.Errorf("basispoints correction did not complete")
			} else {
				output, _ := response["output"].([]any)
				for _, raw := range output {
					item, _ := raw.(object)
					if isTool(item) {
						delete(pending, text(item["call_id"])+"\x00"+text(item["id"]))
					}
				}
				if len(pending) != 0 {
					terminalError = fmt.Errorf("basispoints correction omitted an original tool item")
				}
			}
			return io.EOF
		}
		return nil
	})
	if err != nil && !errors.Is(err, io.EOF) {
		return response, err
	}
	if terminalError != nil {
		return response, terminalError
	}
	if response == nil {
		return nil, io.ErrUnexpectedEOF
	}
	return response, nil
}

// A correction chooses a declared raw transport, not replacement source text.
// Bind its code field to the original bytes before checking the whole batch.
// Valid operations and named JSON envelopes are never rebound.
// Clone corrected calls so replay records exactly what the client receives,
// without rewriting the model response used by the continuation.
func (b *Bridge) restoreRawToolPayloads(original, corrected []object) ([]object, error) {
	check := *b
	check.replay = nil
	result := append([]object(nil), corrected...)
	for i, native := range original {
		if _, err := check.translateCall(copyToolObject(native)); err == nil {
			continue
		}
		code, ok := transportArguments(native)["code"].(string)
		if !ok {
			continue
		}
		if envelope, err := decodeTransportEnvelope(code); err == nil {
			if name, nameErr := envelopeName(envelope); nameErr == nil && name != "" {
				continue
			}
		}
		args := transportArguments(corrected[i])
		summary := text(args["summary"])
		if (!strings.HasPrefix(summary, customTransportPrefix) && !strings.HasPrefix(summary, functionCodeTransportPrefix)) || args["code"] == code {
			continue
		}
		boundArgs := make(object, len(args))
		for key, value := range args {
			boundArgs[key] = value
		}
		boundArgs["code"] = code
		encoded, err := json.Marshal(boundArgs)
		if err != nil {
			return nil, fmt.Errorf("basispoints cannot bind the original tool payload: %w", err)
		}
		bound := make(object, len(corrected[i]))
		for key, value := range corrected[i] {
			bound[key] = value
		}
		bound["arguments"] = string(encoded)
		result[i] = bound
	}
	return result, nil
}
