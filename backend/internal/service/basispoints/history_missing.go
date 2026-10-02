package basispoints

import "fmt"

// MissingToolHistoryError never guesses an operation from its result or another
// tenant's cache. Its index lets the client resend the precise missing history.
type MissingToolHistoryError struct{ Index int }

func (e *MissingToolHistoryError) Path() string { return fmt.Sprintf("input[%d].call_id", e.Index) }
func (e *MissingToolHistoryError) Error() string {
	return "basispoints original tool item is unavailable for this tool result at " + e.Path() + "; resend the complete original call and result history, or start a new conversation with a saved summary"
}
