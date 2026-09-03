"""
Build web/index.html by embedding web/model_export.json into web/template.html
(replacing the __MODEL_DATA_JSON__ placeholder). The result is a single
self-contained HTML file — open it directly in a browser, no server needed.

Usage:
    python src/build_web_page.py

Run this after python src/export_web_demo_data.py to refresh the page with
a newly retrained model.
"""
from __future__ import annotations

import json

TEMPLATE_PATH = "web/template.html"
DATA_PATH = "web/model_export.json"
OUT_PATH = "web/index.html"


def main() -> None:
    with open(DATA_PATH) as f:
        data = json.load(f)

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = f.read()

    if "__MODEL_DATA_JSON__" not in template:
        raise SystemExit(f"{TEMPLATE_PATH} is missing the __MODEL_DATA_JSON__ placeholder")

    out = template.replace("__MODEL_DATA_JSON__", json.dumps(data))
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(out)

    print(f"Wrote {OUT_PATH} ({len(out):,} bytes, {len(data['teams'])} teams)")


if __name__ == "__main__":
    main()
