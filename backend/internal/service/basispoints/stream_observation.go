package basispoints

// StreamStage is an allowlisted lifecycle marker, never upstream content.
type StreamStage string

const (
	StreamMessageCompleted    StreamStage = "message_completed"
	StreamFirstEvent          StreamStage = "first_event"
	StreamFirstOutput         StreamStage = "first_output"
	StreamToolReady           StreamStage = "tool_ready"
	StreamToolEmitted         StreamStage = "tool_emitted"
	StreamCompactionStarted   StreamStage = "compaction_started"
	StreamCompactionCompleted StreamStage = "compaction_completed"
	StreamUpstreamCompleted   StreamStage = "upstream_completed"
	StreamValidationStarted   StreamStage = "validation_started"
	StreamValidationCompleted StreamStage = "validation_completed"
	StreamRepairStarted       StreamStage = "repair_started"
	StreamRepairCompleted     StreamStage = "repair_completed"
)

// ObserveLifecycle must be installed before streaming. The callback is
// synchronous, must be bounded, and must not mutate or dispatch tool calls.
func (b *Bridge) ObserveLifecycle(fn func(StreamStage)) { b.observeLifecycle = fn }

func (b *Bridge) observe(stage StreamStage) {
	if b.observeLifecycle != nil {
		b.observeLifecycle(stage)
	}
}
