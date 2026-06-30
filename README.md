
# Evidence-k

Evidence-k is a benchmarking and calibration framework for measuring the optimal amount of evidence an LLM should receive during inference.

The core assumption is simple: more context is not always better. For a given model, task type, evidence format, and control structure, there is often a measurable saturation point where additional evidence no longer improves reliability and may instead increase drift, hallucination, constraint violations, or cost.

Evidence-k estimates this point as a model-specific and task-specific k* value.

The resulting `k_profile.json` files can be used by routers, RAG systems, memory layers, and inference-time control architectures to decide how many evidence fragments or state slices should be injected into each model call.
