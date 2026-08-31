"""Small hand-written catalogs for the retrieval tests.

Never the real 50k file — these are shaped so each test can reason about the
expected answer by hand.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


def product(
    parent_asin: str,
    *,
    title: str = "product",
    features: list[str] | None = None,
    details: dict | None = None,
    description: list[str] | None = None,
    categories: list[str] | None = None,
    rating_number: int = 0,
    average_rating: float = 4.0,
    price: object = None,
    store: str = "Example",
) -> dict:
    return {
        "parent_asin": parent_asin,
        "title": title,
        "features": features or [],
        "details": details or {},
        "description": description or [],
        "categories": categories or ["Clothing", "Shoes"],
        "store": store,
        "average_rating": average_rating,
        "rating_number": rating_number,
        "price": price,
    }


@contextmanager
def catalog_file(rows: list[dict]):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "catalog.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        yield path


def normalised_product(
    parent_asin: str,
    *,
    category: str = "Clothing",
    audience: str = "unisex",
    brand: str = "example",
    material: list[str] | None = None,
    color: list[str] | None = None,
    feature: list[str] | None = None,
    style: list[str] | None = None,
    use_case: list[str] | None = None,
    region: str = "unknown",
    price: float | None = None,
) -> dict:
    """One row in the faceted normalised catalog shape."""
    return {
        "parent_asin": parent_asin,
        "category": category,
        "audience": audience,
        "brand": brand,
        "material": material or [],
        "color": color or [],
        "feature": feature or [],
        "style": style or [],
        "use_case": use_case or [],
        "price": price,
        "region": region,
    }


@contextmanager
def normalised_catalog_file(rows: list[dict], raw_rows: list[dict] | None = None):
    """Write a normalised catalog and, optionally, its row-aligned raw sidecar.

    Yields the normalised path. When ``raw_rows`` is given it is written as
    ``catalog.jsonl`` beside it, which is exactly where ``DualTrackRetriever``
    and ``CatalogIndex`` probe for the numeric sidecar.
    """
    with TemporaryDirectory() as directory:
        path = Path(directory) / "catalog_normalised.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        if raw_rows is not None:
            sidecar = Path(directory) / "catalog.jsonl"
            sidecar.write_text(
                "".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8"
            )
        yield path


# A catalog with two distinct breadcrumbs and a clear popularity ordering.
BOOTS_AND_NECKLACES = [
    product(
        "BOOT1",
        title="Waterproof hiking boot",
        features=["waterproof leather upper", "Vibram outsole"],
        details={"Material": "leather"},
        categories=["Clothing, Shoes & Jewelry", "Shoes", "Hiking Boots"],
        rating_number=500,
        price=89.0,
    ),
    product(
        "BOOT2",
        title="Trail boot",
        features=["leather upper that is waterproof", "lightweight"],
        details={"Material": "suede"},
        categories=["Clothing, Shoes & Jewelry", "Shoes", "Hiking Boots"],
        rating_number=50,
        price=59.0,
    ),
    product(
        "BOOT3",
        title="Plain boot",
        features=["canvas upper"],
        categories=["Clothing, Shoes & Jewelry", "Shoes", "Hiking Boots"],
        rating_number=5,
        price="—",
    ),
    product(
        "NECK1",
        title="Triple moon necklace",
        features=["Triple Moon Pentagram Symbol"],
        details={"Material": "alloy"},
        categories=["Clothing, Shoes & Jewelry", "Jewelry", "Necklaces"],
        rating_number=200,
        price=19.99,
    ),
    product(
        "NECK2",
        title="Simple chain necklace",
        features=["stainless steel chain"],
        categories=["Clothing, Shoes & Jewelry", "Jewelry", "Necklaces"],
        rating_number=1000,
        price=19.99,
    ),
]


# The same five products in the normalised faceted shape: a distinct brand per
# row and a clear audience / category split. Row order matches the raw sidecar
# below so CatalogIndex can verify the alignment.
NORMALISED_BOOTS_AND_NECKLACES = [
    normalised_product("BOOT1", category="Shoes", audience="men",
                       brand="vibram", material=["leather"],
                       feature=["waterproof"], price=89.0),
    normalised_product("BOOT2", category="Shoes", audience="men",
                       brand="trailco", material=["suede"],
                       feature=["lightweight"], price=59.0),
    normalised_product("BOOT3", category="Shoes", audience="unisex",
                       brand="plainco", material=["canvas"]),
    normalised_product("NECK1", category="Jewelry", audience="women",
                       brand="moonco", material=["alloy"],
                       style=["boho"], price=19.99),
    normalised_product("NECK2", category="Jewelry", audience="women",
                       brand="chainco", material=["stainless steel"],
                       price=19.99),
]

# Raw sidecar, row-aligned with NORMALISED_BOOTS_AND_NECKLACES by parent_asin.
# It carries the lexical text (title / features) the faceted table drops, plus
# the review counts -- these mirror BOOTS_AND_NECKLACES so the numeric tests
# read the same way.
SIDECAR_BOOTS_AND_NECKLACES = [
    product("BOOT1", title="Waterproof hiking boot",
            features=["waterproof leather upper", "Vibram outsole"],
            categories=["Clothing, Shoes & Jewelry", "Shoes", "Hiking Boots"],
            rating_number=500, average_rating=4.5, price=89.0),
    product("BOOT2", title="Trail boot",
            features=["leather upper that is waterproof", "lightweight"],
            categories=["Clothing, Shoes & Jewelry", "Shoes", "Hiking Boots"],
            rating_number=50, average_rating=4.0, price=59.0),
    product("BOOT3", title="Plain boot", features=["canvas upper"],
            categories=["Clothing, Shoes & Jewelry", "Shoes", "Hiking Boots"],
            rating_number=5, average_rating=3.0, price="—"),
    product("NECK1", title="Triple moon necklace",
            features=["Triple Moon Pentagram Symbol"],
            categories=["Clothing, Shoes & Jewelry", "Jewelry", "Necklaces"],
            rating_number=200, average_rating=4.8, price=19.99),
    product("NECK2", title="Simple chain necklace",
            features=["stainless steel chain"],
            categories=["Clothing, Shoes & Jewelry", "Jewelry", "Necklaces"],
            rating_number=1000, average_rating=4.2, price=19.99),
]
