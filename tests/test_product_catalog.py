from pathlib import Path

from src.product_catalog import build_catalog, find_product, format_catalog, load_catalog


def test_build_catalog_includes_descriptions_images_and_specs(tmp_path):
    catalog_path = tmp_path / "product_catalog.json"

    products = build_catalog(Path("data/productspecification.md"), catalog_path)

    assert catalog_path.exists()
    assert len(products) >= 8
    wm_a_plus = find_product("WM 1325 A+ specification", products)
    assert wm_a_plus is not None
    assert wm_a_plus["image"] == "data/images/wm1325A+.png"
    assert "premium industrial CNC router" in wm_a_plus["description"]
    assert any("AC Servo Motor" in item for item in wm_a_plus["technical_specifications"])
    assert any("Auto Homing" in item for item in wm_a_plus["unique_features"])

    loaded = load_catalog(catalog_path)
    assert loaded[0]["name"] == products[0]["name"]


def test_format_catalog_outputs_catalog_sections():
    product = {
        "name": "Demo Machine",
        "description": "A useful machine.",
        "image": "data/images/demo.png",
        "technical_specifications": ["Spec A"],
        "toolbox_contents": ["Tool A"],
        "unique_features": ["Feature A"],
    }

    answer = format_catalog(product)

    assert "Product catalog: Demo Machine" in answer
    assert "Image: data/images/demo.png" in answer
    assert "Overview:" in answer
    assert "Technical specifications:" in answer
    assert "- Tool A" in answer
