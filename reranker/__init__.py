"""Candidate shrinking (Stage B) + LLM rerank (Stage C).

`reranker.reranker` is self-contained and importable on its own.
`reranker.adapter` is the pipeline-facing bridge and imports from `starter`,
so it is NOT re-exported here - import it directly to keep that dependency
out of anyone who only wants shrink_pool.
"""

from reranker.reranker import CONFIG, SYNONYMS, llm_rerank, load_env, shrink_pool

__all__ = ["CONFIG", "SYNONYMS", "llm_rerank", "load_env", "shrink_pool"]
