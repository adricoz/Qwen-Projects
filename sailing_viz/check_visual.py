"""Visual + numeric check of index.html in a headless Edge driven by Playwright (software WebGL).

Setup (once):  python -m venv pwenv && pwenv/Scripts/pip install playwright   (uses the installed Edge, no download)
Run:           pwenv/Scripts/python check_visual.py . out_dir
Prints world positions of key parts for sign checks and writes 5 screenshots into out_dir.
"""
import http.server
import os
import sys
import threading
import time
from functools import partial

from playwright.sync_api import sync_playwright

WORK = sys.argv[1]
OUT = sys.argv[2]
PORT = 8766

handler = partial(http.server.SimpleHTTPRequestHandler, directory=WORK)
handler.log_message = lambda *a, **k: None
httpd = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
httpd.daemon_threads = True
threading.Thread(target=httpd.serve_forever, daemon=True).start()

WORLD = """(name) => { let o = null; const q = name.replace(/ /g, '_');
  window.VIZ.scene.traverse(x => { if (!o && (x.name === name || x.name === q || x.name.includes(name) || x.name.includes(q))) o = x; });
  if (!o) return null; o.updateWorldMatrix(true, true);
  const c = new THREE.Vector3(); new THREE.Box3().setFromObject(o).getCenter(c); return [c.x, c.y, c.z]; }"""

errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True,
                                args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--disable-gpu",
                                      "--no-sandbox"])
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.on("console", lambda m: errors.append(m.text) if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: errors.append("PAGEERROR " + str(e)))
    t0 = time.time()
    page.goto(f"http://127.0.0.1:{PORT}/index.html", wait_until="load", timeout=120_000)
    page.wait_for_function("document.getElementById('loading').style.display === 'none'", timeout=300_000)
    print(f"model loaded after {time.time() - t0:.0f} s")
    print("mast-like node names:", page.evaluate("() => { const r = []; window.VIZ.scene.traverse(x => { if (/mast|boom/i.test(x.name)) r.push(x.name); }); return r.slice(0, 6); }"))
    page.wait_for_timeout(2000)
    page.screenshot(path=os.path.join(OUT, "pw_default.png"))
    def world(name):
        return page.evaluate(WORLD, name)

    def setp(**kw):
        for k, v in kw.items():
            page.evaluate("([k, v]) => window.VIZ.setParam(k, v)", [k, v])
        page.wait_for_timeout(300)

    # --- numeric checks -------------------------------------------------
    setp(heel=0, trim=0, yaw=0, rudder=0, cant=0, rake=0, foilExtension=1, rideHeight=0)
    tip1 = world("foil_tip")
    setp(foilExtension=0)
    tip0 = world("foil_tip")
    print("foil_tip y at extension 1 / 0:", round(tip1[1], 2), round(tip0[1], 2), "-> raised" if tip0[1] > tip1[1] else "-> LOWERED (bug)")
    setp(foilExtension=1)
    elev = world("elevator")
    print("elevator world pos:", [round(v, 2) for v in elev], "(expected x ~ -3.85, y ~ -1.2)")
    setp(heel=25)
    mast = world("Yacht sloop finite aluminium mast")
    print("mast centre z at heel +25:", round(mast[2], 2), "(expected > 0 = starboard)")
    setp(heel=0, trim=10)
    bow = world("arrow_heading")
    print("bow (heading arrow origin) y at trim +10:", round(bow[1], 2), "(expected > 0 = bow up)")
    setp(trim=0, rudder=25)
    rud = world("Yacht finite transom rudder blade")
    print("rudder blade centre z at rudder +25:", round(rud[2], 3), "(expected > 0: trailing edge to starboard)")
    setp(rudder=0, twa=60)
    boom = world("Yacht sloop attached boom")
    gen = world("gennaker")
    print("boom centre z at TWA +60:", round(boom[2], 2), "(expected < 0 = port)")
    setp(sailConfig="main_gennaker")
    print("gennaker pivot world:", [round(v, 2) for v in world("gennaker_pivot")], "(expected ~ 5.5, 0.8, 0)")
    setp(twa=-60)
    boom = world("Yacht sloop attached boom")
    print("boom centre z at TWA -60:", round(boom[2], 2), "(expected > 0 = starboard)")

    # --- screenshots ----------------------------------------------------
    page.click("#btn-reset")
    page.wait_for_timeout(500)
    setp(twa=-130, sailConfig="main_spinnaker", heel=25, cant=30, foilExtension=1, rideHeight=1.2, rudder=20, boatSpeed=30)
    page.evaluate("window.VIZ.highlight('rudder')")
    page.click("#cam-stern")
    page.wait_for_timeout(2500)
    page.screenshot(path=os.path.join(OUT, "pw_config2.png"))
    print("readout config2:", page.evaluate("['readout-awa','readout-aws','readout-vmg']"
                                            ".map(i => document.getElementById(i).textContent)"))
    page.click("#cam-side")
    setp(twa=90, sailConfig="main_gennaker", heel=-10, foilExtension=0, rideHeight=0)
    page.wait_for_timeout(2500)
    page.screenshot(path=os.path.join(OUT, "pw_side_retracted.png"))
    page.click("#cam-top")
    setp(twa=60, sailConfig="main_genoa", leeway=12, heel=0, foilExtension=1)
    page.wait_for_timeout(2500)
    page.screenshot(path=os.path.join(OUT, "pw_top.png"))
    # glossary click
    page.fill("#glossary-search", "dérive")
    page.click(".term[data-term='leeway']")
    page.wait_for_timeout(800)
    print("definition box:", page.inner_text("#definition-box")[:140].replace("\n", " | "))
    page.screenshot(path=os.path.join(OUT, "pw_glossary.png"))
    browser.close()
real = [e for e in errors if "GPU stall" not in e and "favicon" not in e]
print("console errors/warnings (excluding GPU stall/favicon):", len(real))
for e in real[:10]:
    print("  ", e[:200])
