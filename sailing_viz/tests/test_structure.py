# -*- coding: utf-8 -*-
"""Structural tests for the 3D sailing-vocabulary visualiser (index.html)."""

import base64
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INDEX = os.path.join(ROOT, "index.html")
GLOSSARY = os.path.join(ROOT, "glossary.json")


def read_text(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def html():
    return read_text(INDEX)


def script_body():
    m = re.search(r"<script>\n(.*?)\n</script>", html(), re.S)
    assert m, "inline application script not found"
    return m.group(1)


def run_node(code):
    tmp = os.path.join(ROOT, "_t_node_check.js")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        r = subprocess.run(["node", tmp], capture_output=True, text=True, cwd=ROOT)
        return r.returncode, r.stdout, r.stderr
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def test_index_html_exists():
    assert os.path.isfile(INDEX)
    h = html()
    assert "sailing-yacht-8-m.glb" in h
    for tag in (
        "three.min.js",
        "OrbitControls.js",
        "GLTFLoader.js",
    ):
        assert tag in h


REQUIRED_IDS = [
    "view", "loading", "glossary", "glossary-search", "definition-box",
    "slider-heel", "slider-trim", "slider-yaw", "slider-leeway",
    "slider-cant", "slider-rake", "slider-foil-extension", "slider-ride-height",
    "slider-rudder", "slider-twa", "slider-tws", "slider-boat-speed",
    "sail-config",
    "cam-3quarter", "cam-side", "cam-top", "cam-stern", "cam-bow",
    "btn-reset",
    "readout-awa", "readout-aws", "readout-vmg", "readout-heading", "readout-course",
    "value-heel", "value-trim", "value-yaw", "value-leeway", "value-cant",
    "value-rake", "value-foil-extension", "value-ride-height", "value-rudder",
    "value-twa", "value-tws", "value-boat-speed",
]


def test_all_control_ids_present():
    h = html()
    for i in REQUIRED_IDS:
        assert 'id="%s"' % i in h, "missing id %s" % i


def test_glossary_block_equals_glossary_json():
    h = html()
    m = re.search(
        r'<script id="glossary-data" type="application/json">(.*?)</script>',
        h, re.S,
    )
    assert m, "glossary JSON block not found"
    inline = json.loads(m.group(1))
    source = json.loads(read_text(GLOSSARY))
    assert inline == source, "glossary block differs from glossary.json"


def test_sliders_match_spec():
    h = html()
    spec = {
        "slider-heel": (-40, 40, 1, -15),
        "slider-trim": (-15, 15, 0.5, 2),
        "slider-yaw": (-180, 180, 1, 0),
        "slider-leeway": (-15, 15, 0.5, 4),
        "slider-cant": (-20, 40, 1, 10),
        "slider-rake": (-5, 10, 0.25, 2),
        "slider-foil-extension": (0, 1, 0.05, 1),
        "slider-ride-height": (0, 1.5, 0.05, 0.6),
        "slider-rudder": (-30, 30, 1, 0),
        "slider-twa": (-180, 180, 1, 45),
        "slider-tws": (0, 40, 1, 15),
        "slider-boat-speed": (0, 40, 0.5, 18),
    }
    for sid, (lo, hi, step, val) in spec.items():
        m = re.search(r'<input[^>]*id="%s"[^>]*>' % sid, h)
        assert m, "slider %s not found" % sid
        tag = m.group(0)
        assert 'type="range"' in tag
        assert 'min="%s"' % lo in tag
        assert 'max="%s"' % hi in tag
        assert 'step="%s"' % step in tag
        assert 'value="%s"' % val in tag


def test_sail_config_options():
    h = html()
    m = re.search(r'<select[^>]*id="sail-config"[^>]*>(.*?)</select>', h, re.S)
    assert m
    for opt in ("main_jib", "main_genoa", "main_gennaker", "main_spinnaker", "main_code_zero"):
        assert 'value="%s"' % opt in m.group(1)


def test_viz_api_defined_in_pure_zone():
    body = script_body()
    assert "function computeApparentWind(" in body
    assert "window.VIZ" in body or "VIZ = {" in body
    assert "if (typeof document !== 'undefined' && typeof THREE !== 'undefined')" in body


def test_inline_script_passes_node_check():
    body = script_body()
    tmp = os.path.join(ROOT, "_t_syntax.js")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(body)
    try:
        r = subprocess.run(["node", "--check", tmp], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def test_compute_apparent_wind_runs_in_node():
    body = script_body()
    probe = (
        "globalThis.window = globalThis;\n"
        + body
        + """
var V = globalThis.VIZ;
var r = V.computeApparentWind(45, 15, 18);
var ok = Math.abs(r.awa - 20.3) < 0.05 && Math.abs(r.aws - 30.5) < 0.05 && Math.abs(r.vmg - 12.7) < 0.05;
console.log("RESULT", ok, JSON.stringify(r));
"""
    )
    rc, out, err = run_node(probe)
    assert rc == 0, err
    assert "RESULT true" in out, out + err


def test_set_param_api_in_node():
    body = script_body()
    probe = (
        "globalThis.window = globalThis;\n"
        + body
        + """
var V = globalThis.VIZ;
V.setParam('twa', 90);
console.log("RESULT", V.getParam('twa') === 90);
"""
    )
    rc, out, err = run_node(probe)
    assert rc == 0, err
    assert "RESULT true" in out


def test_standalone_build():
    sys.path.insert(0, ROOT)
    import build_standalone as bs

    bs.main()
    out = os.path.join(ROOT, "sailing_viz_standalone.html")
    assert os.path.isfile(out)
    h = read_text(out)
    m = re.search(r'window\.MODEL_B64 = "([A-Za-z0-9+/=]+)"', h)
    assert m, "MODEL_B64 not injected"
    raw = base64.b64decode(m.group(1))
    with open(os.path.join(ROOT, "assets", "sailing-yacht-8-m.glb"), "rb") as f:
        assert raw == f.read(), "inlined base64 differs from the GLB file"
    assert h.count('<script>window.MODEL_B64') == 1
