"""Hidden acceptance tests for the sailing_viz task (structure, glossary identity, JS syntax, apparent wind
against an independent vector reference, standalone build). They never import the delivered page; they parse it."""
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

WORK = Path(os.environ["QWEN_AGENT_WORKDIR"])
HERE = Path(__file__).resolve().parent
INDEX = WORK / "index.html"
GLB = WORK / "assets" / "sailing-yacht-8-m.glb"
GLB_SHA = "ca741998d833804a73a747aa28b04912a6e431937cd3372cc7508ea65462850e"

SLIDERS = {
    "slider-heel": (-40, 40, -15), "slider-trim": (-15, 15, 2), "slider-yaw": (-180, 180, 0),
    "slider-leeway": (-15, 15, 4), "slider-cant": (-20, 40, 10), "slider-rake": (-5, 10, 2),
    "slider-foil-extension": (0, 1, 1), "slider-ride-height": (0, 1.5, 0.6), "slider-rudder": (-30, 30, 0),
    "slider-twa": (-180, 180, 45), "slider-tws": (0, 40, 15), "slider-boat-speed": (0, 40, 18),
}
OTHER_IDS = ["sail-config", "cam-3quarter", "cam-side", "cam-top", "cam-stern", "cam-bow", "btn-reset",
             "readout-awa", "readout-aws", "readout-vmg", "readout-heading", "readout-course",
             "definition-box", "glossary-search", "glossary-data"]
SAIL_OPTIONS = ["main_jib", "main_genoa", "main_gennaker", "main_spinnaker", "main_code_zero"]
CDN = ["three.js/r128/three.min.js", "three@0.128.0/examples/js/controls/OrbitControls.js",
       "three@0.128.0/examples/js/loaders/GLTFLoader.js"]


def html():
    return INDEX.read_text(encoding="utf-8")


def tag_attrs(doc, elem_id):
    m = re.search(r"<(\w+)\b([^>]*\bid\s*=\s*[\"']%s[\"'][^>]*)>" % re.escape(elem_id), doc)
    assert m, f"element id={elem_id} missing"
    return m.group(1), dict(re.findall(r"(\w[\w-]*)\s*=\s*[\"']([^\"']*)[\"']", m.group(2)))


def inline_scripts(doc):
    out = []
    for m in re.finditer(r"<script\b([^>]*)>(.*?)</script>", doc, re.S | re.I):
        attrs = m.group(1)
        if re.search(r"\bsrc\s*=", attrs) or "application/json" in attrs:
            continue
        if m.group(2).strip():
            out.append(m.group(2))
    return out


def app_script(doc):
    scripts = inline_scripts(doc)
    assert scripts, "no inline application script"
    return "\n".join(scripts)


def test_files_exist():
    for f in ["index.html", "build_standalone.py", "serve.bat", "README.md", "tests/test_structure.py"]:
        assert (WORK / f).is_file(), f
    assert "http.server" in (WORK / "serve.bat").read_text(encoding="utf-8", errors="replace")


def test_model_untouched():
    assert hashlib.sha256(GLB.read_bytes()).hexdigest() == GLB_SHA


def test_cdn_scripts_only():
    doc = html()
    srcs = re.findall(r"<script[^>]*\bsrc\s*=\s*[\"']([^\"']+)[\"']", doc)
    assert len(srcs) == 3, srcs
    for want in CDN:
        assert any(want in s for s in srcs), want


def test_sliders():
    doc = html()
    for sid, (lo, hi, default) in SLIDERS.items():
        tag, a = tag_attrs(doc, sid)
        assert tag.lower() == "input" and a.get("type") == "range", sid
        assert float(a["min"]) == lo and float(a["max"]) == hi, (sid, a)
        assert float(a["value"]) == default, (sid, a.get("value"))
        tag_attrs(doc, sid.replace("slider-", "value-"))


def test_other_ids_and_options():
    doc = html()
    for i in OTHER_IDS:
        tag_attrs(doc, i)
    tag, _ = tag_attrs(doc, "sail-config")
    assert tag.lower() == "select"
    sel = re.search(r"<select\b[^>]*id\s*=\s*[\"']sail-config[\"'][^>]*>(.*?)</select>", doc, re.S)
    values = re.findall(r"value\s*=\s*[\"']([^\"']+)[\"']", sel.group(1))
    assert values == SAIL_OPTIONS, values


def test_glossary_block_is_verbatim():
    doc = html()
    m = re.search(r"<script\b[^>]*id\s*=\s*[\"']glossary-data[\"'][^>]*>(.*?)</script>", doc, re.S)
    assert m and "application/json" in m.group(0)
    got = json.loads(m.group(1))
    ref = json.loads((HERE / "glossary_ref.json").read_text(encoding="utf-8"))
    assert got == ref, "glossary block differs from glossary.json (entries, order or text)"


def test_glossary_groups_in_panel():
    doc = html()
    for title in ["Attitude du bateau", "Foils et appendices", "Voiles et gréement", "Vent et allures", "Coque"]:
        norm = unicodedata.normalize("NFC", title)
        assert norm in unicodedata.normalize("NFC", doc), title


def test_js_syntax():
    doc = html()
    with tempfile.TemporaryDirectory() as d:
        for k, src in enumerate(inline_scripts(doc)):
            p = Path(d) / f"s{k}.js"
            p.write_text(src, encoding="utf-8")
            r = subprocess.run(["node", "--check", str(p)], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[:2000]


def _reference_apparent(twa, tws, v):
    """Independent reference: explicit 2-D vector sum, wind FROM angle twa measured from the bow, + = starboard."""
    a = math.radians(twa)
    # unit vector pointing TOWARDS the wind source: (cos a, sin a) in (forward, starboard)
    src_f, src_s = math.cos(a), math.sin(a)
    # true wind velocity blows away from the source; apparent = true wind - boat velocity (boat along +forward)
    aw_f = -tws * src_f - v
    aw_s = -tws * src_s
    aws = math.hypot(aw_f, aw_s)
    # apparent source direction = -AW
    awa = math.degrees(math.atan2(aw_s * -1, aw_f * -1))
    vmg = v * math.cos(a)
    return awa, aws, vmg


def _run_node_apparent(cases):
    doc = html()
    js = app_script(doc)
    prog = (
        "globalThis.window = globalThis;\n" + js + "\n"
        "const cases = " + json.dumps(cases) + ";\n"
        "console.log(JSON.stringify(cases.map(c => window.VIZ.computeApparentWind(c[0], c[1], c[2]))));\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "run.js"
        p.write_text(prog, encoding="utf-8")
        r = subprocess.run(["node", str(p)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, "script must run in Node with only window defined:\n" + r.stderr[:2000]
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.startswith("[")]
    return json.loads(lines[-1])


def test_apparent_wind_values():
    cases = [[45, 15, 18], [-45, 15, 18], [0, 10, 5], [180, 20, 10], [90, 12, 0], [120, 18, 25], [-150, 8, 3]]
    got = _run_node_apparent(cases)
    for c, g in zip(cases, got):
        awa, aws, vmg = _reference_apparent(*c)
        assert abs(g["aws"] - aws) < 1e-6, (c, g, aws)
        assert abs(g["vmg"] - vmg) < 1e-6, (c, g, vmg)
        if aws > 1e-9:
            d = (g["awa"] - awa + 180) % 360 - 180
            assert abs(d) < 1e-6, (c, g, awa)


def test_apparent_wind_properties():
    got = _run_node_apparent([[60, 15, 12], [-60, 15, 12], [60, 15, 0], [179, 15, 30]])
    assert abs(got[0]["awa"] + got[1]["awa"]) < 1e-9 and abs(got[0]["aws"] - got[1]["aws"]) < 1e-9
    assert abs(got[2]["awa"] - 60) < 1e-9 and abs(got[2]["aws"] - 15) < 1e-9  # no boat speed: apparent = true
    assert 0 < got[0]["awa"] < 60  # moving forward brings the apparent wind forward
    assert got[3]["aws"] > 14  # faster than the wind downwind: apparent wind from ahead


def test_viz_api_names():
    js = app_script(html())
    for name in ["setParam", "getParam", "applyState", "highlight", "TERMS", "computeApparentWind", "MODEL_B64",
                 "loader.parse", "foil_port", "foil_starboard", "foil_shaft", "foil_tip", "elevator", "bowsprit",
                 "arrow_heading", "arrow_course", "arrow_true_wind", "arrow_apparent_wind", "ArrowHelper",
                 "OrbitControls", "GLTFLoader", "resize"]:
        assert name in js, name


def test_build_standalone():
    r = subprocess.run([sys.executable, "-I", str(WORK / "build_standalone.py")], cwd=WORK,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    out = WORK / "sailing_viz_standalone.html"
    assert out.is_file()
    txt = out.read_text(encoding="utf-8")
    import base64
    b64 = base64.b64encode(GLB.read_bytes()).decode()
    assert "window.MODEL_B64" in txt and b64[:200] in txt and b64[-200:] in txt
    assert txt.index("MODEL_B64") < txt.index("computeApparentWind")
    assert len(txt) > 9_000_000
