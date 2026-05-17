# Contamination Audit Report

Accepted items: 4258

Methods actually run:
- Exact normalized prompt/final-answer matching against loaded benchmark text.
- Thai/English whitespace-normalized text matching.
- Character 5-gram Jaccard near-duplicate scoring.
- SimHash similarity on candidates whose character 5-gram Jaccard reached 0.25 or higher.
- Numeric-structure review labels assigned manually for math/coding items.
- Problem-statement review for coding items against LiveCodeBench task families.

Methods not run:
- Neural embedding similarity: status = not_run; max = not run.

Loaded blacklist notes:
- loaded cached math-ai/aime24: 30 rows
- loaded cached math-ai/aime25: 30 rows
- loaded cached math-ai/math500: 500 rows
- loaded cached typhoon-ai/ifeval-th: 215 rows
- loaded cached ThaiLLM-Leaderboard/mt-bench-thai: 91 rows
- loaded cached iapp/openthaieval cached parquet: 1232 rows
- sampled cached typhoon-ai/livecodebench-th public JSONL: 120 rows from large public file

Important limitation: the full LiveCodeBench-TH private-test payload is large, and this run did not download or store the full private-test archive. The audit used public schema/source review and lighter benchmark text loads. A production run should run the same items through a complete offline protected registry.

Max character 5-gram Jaccard observed: 0.2667
SimHash status: run_on_ngram_ge_0.25_candidates
Max SimHash similarity observed: 0.7500

Decision threshold used in this partial run:
- reject exact normalized match
- reject char 5-gram Jaccard >= 0.35
- manually reject any story skeleton or coding task that resembles official/public benchmark tasks
