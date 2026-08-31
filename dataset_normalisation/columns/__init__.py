"""One module per output column of the normalised catalog table.

Each exposes a single pure function over a raw catalog row and holds its own
closed vocabulary. Orchestration lives in dataset_normalisation/pipeline.py; the schema, enums
and data-quality notes are in dataset_normalisation/REFORMATTER_README.md.

    audience.py         audience()          -> (str, source)
    price.py            parse_price()       -> float | None
    category.py         product_family()    -> (str, source)
    color.py            extract_colors()    -> (list[str], source)
    feature.py          extract_features()  -> list[str]
    material.py         extract_materials() -> list[str]
    region.py           region()            -> (str, source)
    style_use_case.py   build_request()     -> Batch API request (LLM)
"""
