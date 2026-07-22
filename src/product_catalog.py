"""Product catalogue loading, parsing, matching, and presentation helpers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _clean(value: str) -> str:
    replacements = {
        "Ã¢â‚¬â€œ": "-",
        "â€“": "-",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
    }
    cleaned = value.strip()
    for broken, replacement in replacements.items():
        cleaned = cleaned.replace(broken, replacement)
    return re.sub(r"\s+", " ", cleaned)


def _items(section: str) -> list[str]:
    values: list[str] = []
    for line in section.splitlines():
        line = _clean(line)
        line = re.sub(r"^(?:\d+(?:[.)]|\s+)|[-*])\s*", "", line)
        if line and not line.lower().startswith(("no.", "feature", "item", "cnc machine features")):
            values.append(line)
    return values


def _product_key(name: str) -> str:
    normalized = _clean(name).lower().replace("+", "plus")
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _load_product_descriptions(description_path: Path | None = None) -> dict[str, str]:
    path = description_path or Path("data/productdescription.md")
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    matches = re.finditer(
        r"product name\s*:\s*(.+?)\s*,?\s*\n?\s*description\s*:\s*(.*?)(?=\n\s*product name\s*:|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return {
        _product_key(match.group(1).strip(" ,")): _clean(match.group(2))
        for match in matches
    }


def parse_product_specifications(markdown: str) -> list[dict[str, Any]]:
    """Parse the supplied product specification markdown into catalogue records."""
    blocks = re.split(r"^-{20,}\s*$", markdown.replace("\r\n", "\n"), flags=re.MULTILINE)
    descriptions = _load_product_descriptions()
    products: list[dict[str, Any]] = []
    for block in blocks:
        name_match = re.search(r"product name\s*:\s*(.+)", block, flags=re.IGNORECASE)
        if not name_match:
            continue
        name = _clean(name_match.group(1).rstrip(","))
        image_match = re.search(r"^image\s*:\s*(.+)$", block, flags=re.IGNORECASE | re.MULTILINE)
        image = _clean(image_match.group(1)).replace("\\", "/") if image_match else ""
        image = re.sub(r"^[A-Za-z]:/[^/]*/", "", image)
        image = image.lstrip("/")
        if image and not image.startswith("data/"):
            image = f"data/{image}"

        feature_match = re.search(
            r"(?:features(?:/technical ?specification)?\s*:\s*|CNC Machine Features\s*)(.*?)(?=Standard Toolbox|Toolbox\s*:|Unique Features|unique features\s*:|\Z)",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        toolbox_match = re.search(
            r"(?:Standard Toolbox|Toolbox)\s*:?\s*(.*?)(?=Unique Features|unique features\s*:|\Z)",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        unique_match = re.search(r"(?:Unique Features|unique features)\s*:?\s*(.*?)\Z", block, flags=re.IGNORECASE | re.DOTALL)
        products.append(
            {
                "id": re.sub(r"[^a-z0-9]+", "-", name.lower().replace("+", "-plus")).strip("-"),
                "name": name,
                "description": descriptions.get(_product_key(name), ""),
                "image": image or None,
                "technical_specifications": _items(feature_match.group(1)) if feature_match else [],
                "toolbox_contents": _items(toolbox_match.group(1)) if toolbox_match else [],
                "unique_features": _items(unique_match.group(1)) if unique_match else [],
            }
        )
    return products


def build_catalog(specification_path: Path, output_path: Path) -> list[dict[str, Any]]:
    products = parse_product_specifications(specification_path.read_text(encoding="utf-8"))
    output_path.write_text(json.dumps({"products": products}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return products


def load_catalog(path: Path) -> list[dict[str, Any]]:
    return list(json.loads(path.read_text(encoding="utf-8")).get("products", []))


def find_product(question: str, products: list[dict[str, Any]]) -> dict[str, Any] | None:
    query = question.lower()
    aliases = {
        "wood drill": "wood drill",
        "band saw": "band saw",
        "side cutter": "side cutter",
        "randha": "randha",
        "planner": "planner",
    }
    for product in products:
        name = str(product["name"]).lower()
        if name in query:
            return product
        if any(alias in query and marker in name for alias, marker in aliases.items()):
            return product
        model = re.search(r"wm\s*(1325|1625|1825)\s*([ab]\+?)(?!\+)", query)
        product_variant = re.search(r"\s([ab](?:\+)?)$", name.lower())
        if model and model.group(1) in name and product_variant and model.group(2) == product_variant.group(1):
            return product
    return None


def format_catalog(product: dict[str, Any]) -> str:
    # The actual image is sent as a real attachment via the response's `images`
    # list (see rag_pipeline.query/query_stream) -- it must not also be quoted
    # as a raw file path in the text answer.
    lines = [f"Product catalog: {product['name']}"]
    if product.get("description"):
        lines.append(f"\nOverview:\n{product['description']}")
    for title, key in (
        ("Technical specifications", "technical_specifications"),
        ("Toolbox contents", "toolbox_contents"),
        ("Unique features", "unique_features"),
    ):
        values = product.get(key) or []
        if values:
            lines.append(f"\n{title}:")
            lines.extend(f"- {value}" for value in values)
    if len(lines) == 1:
        lines.append("\nDetailed technical specifications have not been supplied for this product yet.")
    return "\n".join(lines)
