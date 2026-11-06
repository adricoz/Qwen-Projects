#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a single-file standalone version of the 3D sailing visualiser.

Reads ``index.html`` and ``assets/sailing-yacht-8-m.glb``, base64-encodes the
model and writes ``sailing_viz_standalone.html`` with an extra
``<script>window.MODEL_B64 = "..."</script>`` inserted just before the
application script. The standalone file opens by double-click (file://).
"""

import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "index.html")
GLB = os.path.join(HERE, "assets", "sailing-yacht-8-m.glb")
OUT = os.path.join(HERE, "sailing_viz_standalone.html")


def main() -> None:
    with open(INDEX, "r", encoding="utf-8") as f:
        html = f.read()
    with open(GLB, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    marker = "<script>\n"  # opening of the (single) inline application script
    idx = html.find(marker)
    if idx == -1:
        raise SystemExit("application <script> not found in index.html")
    inject = '<script>window.MODEL_B64 = "%s";</script>\n' % b64
    out = html[:idx] + inject + html[idx:]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out)
    print("wrote %s (%.2f MB)" % (OUT, len(out) / 1e6))


if __name__ == "__main__":
    main()
