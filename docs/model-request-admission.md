# Model Request Admission

Models with one or a few inference slots should not receive an unbounded
burst of simultaneous HTTP requests. Set these fields in the model's
`[[models]]` entry:

```toml
max_concurrency = 1
max_queue_size = 64
queue_timeout = 180.0
```

The default `max_concurrency = 0` preserves unrestricted traffic. These
settings are application-side controls; they do not change the backend's
slot count or offload settings.

The queue is FIFO and shared across the Bot and WebUI event loops for the
same provider name and model identifier. Model aliases with that same pair
share a queue. Other models and providers are independent. Configure aliases
consistently; a new request applies its configured concurrency limit to the
shared queue.

Queue waiting has its own bounded deadline. The task's existing
`hard_timeout` starts only after admission and still bounds the entire
model attempt, including provider retries. The provider HTTP timeout remains
an additional, independent limit.

Cancellation or timeout returns any granted slot, including cancellation
while waking a waiter on another event loop. A full queue or expired queue
deadline raises a distinct error without sending an HTTP request or
incrementing provider-attempt counters. Failure snapshots use
`model.queue_full` or `model.queue_timeout`, rather than falsely reporting
inference timeouts. Existing candidate-model switching still applies.

Start with the backend's actual slot count. Size the queue and its deadline
for the acceptable background-work age, not for unlimited backlog. GPU
offload can shorten inference, but does not replace admission control.

## Output Budget Failures

Responses API replies marked incomplete because of `max_output_tokens`
raise `OutputTokenLimitError`. Streaming and non-streaming requests preserve
the failed response for diagnostics and never accept truncated structured
output. The orchestrator does not repeat the same model request with the
same exhausted budget; existing candidate-model selection remains available.

This does not silently increase token limits or fix an undersized workload
budget. Inspect the failed task's explicit output limit and the model's
reasoning requirements before changing either. Other parsing failures retain
their existing retry policy.
