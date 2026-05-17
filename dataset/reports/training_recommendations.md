# Training Recommendations

This partial shard is suitable as a pilot-quality sanity set, not as a full post-training corpus.

Recommendations for Qwen/Qwen3.6-35B-A3B:
- Use model-agnostic ChatML-style message records without vendor-specific tokens.
- Keep math and coding answer extraction strict; Qwen's model card recommends standardized final-answer formats for benchmarking.
- Evaluate Thai math/coding deltas per family before scaling generation.
- For agentic items, preserve citation discipline and avoid training on unsupported claims.

Recommendations for Gemma 4:
- Keep outputs concise enough for instruction-tuned variants while retaining reasoning needed for Thai math and code tasks.
- Use JSON/schema-constrained items to exploit Gemma 4's stated structured-output and agentic workflow strengths.
- Separate deterministic IF data from open-ended MT-Bench-style preference data.

Next data to generate:
- More hard Thai geometry and combinatorics with diagram-free wording.
- Larger Thai debugging/self-repair set with mutation tests.
- Thai-local OpenThaiEval-style professional questions with official sources.
- More adversarial IFEval/IFBench Thai constraints, including Buddhist Era dates and Thai numeral constraints.
