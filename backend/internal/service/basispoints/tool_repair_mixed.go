package basispoints

// Adapted from ranxi2001/sub2api 4373ac322. Only mixed batches restore already
// validated operations; serial correction retains its stricter existing rule.
func (b *Bridge) restoreValidatedMixedOperations(original, corrected []object) []object {
	if len(original) != len(corrected) || len(original) < 2 {
		return corrected
	}
	check := *b
	check.replay = nil
	before := make([]object, len(original))
	valid := 0
	for i, item := range original {
		if translated, err := check.translateCall(copyToolObject(item)); err == nil {
			before[i] = translated
			valid++
		}
	}
	if valid == 0 || valid == len(original) {
		return corrected
	}
	result := append([]object(nil), corrected...)
	for i, item := range before {
		if item == nil {
			continue
		}
		after, err := check.translateCall(copyToolObject(corrected[i]))
		if err != nil || item["type"] != after["type"] || item["name"] != after["name"] || text(item["namespace"]) != text(after["namespace"]) {
			continue
		}
		bound := copyToolObject(original[i])
		bound["id"], bound["call_id"] = corrected[i]["id"], corrected[i]["call_id"]
		result[i] = bound
	}
	return result
}
