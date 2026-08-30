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
