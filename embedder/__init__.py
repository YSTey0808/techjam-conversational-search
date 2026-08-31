"""Embeds the catalog for vector retrieval.

    build_embedding_sample.py   pick the sampled rows
    variants.py                 what text each variant embeds (V1 facets, V2 full)
    embed.py                    encode a variant with a registered model
    compare.py                  turn-1 retrieval scores for each variant

Entry point: python3 -m embedder.embed build --variant v2
"""
