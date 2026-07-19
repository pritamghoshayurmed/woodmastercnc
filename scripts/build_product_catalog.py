from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.product_catalog import build_catalog


if __name__ == "__main__":
    build_catalog(root / "data" / "productspecification.md", root / "data" / "product_catalog.json")
