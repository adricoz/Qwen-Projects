# Task: interactive 3D sailing-vocabulary visualiser (Three.js, real yacht model)

Build a web page `index.html` showing a 3D sailing yacht loaded from `assets/sailing-yacht-8-m.glb` (already in
the workdir, CC0, do not modify it). Sliders move the boat, its sails and added foils in real time so a French-speaking
engineer can LEARN the English maritime vocabulary (heel, trim, yaw, cant, rake, leeway, sail names, wind angles...).
Every term is shown as "English term (terme français)". UI text in French; code, comments and identifiers in English.

The page is served by `serve.bat` (`python -m http.server 8765`, open `http://localhost:8765/`). A browser cannot
fetch a GLB from `file://`, so also deliver `build_standalone.py` (see below) which inlines the model as base64 into
`sailing_viz_standalone.html`, a single file that opens by double-click.

## Libraries (exactly these, global `THREE`, no ES modules / import maps, no other library)

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
```

`THREE.OrbitControls` and `THREE.GLTFLoader` are then globals. Everything else is inline in `index.html`:
one `<style>`, the glossary JSON block, one application `<script>`.

## The model (read this carefully, the geometry is baked in world coordinates)

`assets/sailing-yacht-8-m.glb`: 170 named meshes under one root node "Sailing yacht, 8 m", NO node transforms
(all vertices are in model coordinates), 132 k triangles, materials included. Model axes: `+x` = forward (bow at
x = 4.07, stern at x = -3.93), `+y` = up (bulb bottom y = 0, hull bottom y = 1.1, deck y = 2.1 to 2.3, mast top y = 12.76),
`z` = lateral, hull half-beam 1.32 (we define `+z` = starboard). The static waterline is at model y = 1.5: put the
loaded model in a group translated by `y = -1.5` so the sea plane at scene y = 0 is the waterline.

Useful mesh names (`THREE.Object3D.name`, use `.includes(substring)` on `scene.traverse`):

| part | name contains |
|---|---|
| hull | `exact8m finite hull` |
| deck | `sheer-following deck` |
| keel / bulb | `swept finite fin keel` / `shaped ballast bulb` |
| rudder | `transom rudder blade` (x -4.07..-3.68, y 0.35..1.31), `rudder post`, `teak tiller` |
| mast | `aluminium mast` (x 0.87..0.97, z 0, y 2.65..12.76) |
| boom | `attached boom` (x -2.68..0.94, y 4.31..4.41) |
| mainsail (cloth + boltropes + seams + eyelets) | every name containing `mainsail` |
| foresail / jib (same) | every name containing `foresail` |
| sheets | `mainsheet`, `jibsheet` |
| rigging | `tensioned forestay`, `tensioned backstay`, `tensioned shroud` |

The sails are flat in the model (z = 0 to 0.27). To trim them you MUST re-parent:
- `mainGroup` = `THREE.Group` at position (0.92, 0, 0) inside the model group; move every mesh whose name contains
  `mainsail`, `attached boom`, `mainsheet` into it with `mesh.position.set(-0.92, 0, 0)`
  (so the geometry keeps its place) and rotate `mainGroup.rotation.y` = boom angle: sails lie to leeward.
- `jibGroup` = `THREE.Group` at (3.75, 0, 0) for every mesh whose name contains `foresail` or `jibsheet`, same trick,
  rotation about y of about 60 % of the boom angle.
- Rudder group at (-3.85, 0, 0) for `transom rudder blade`, `rudder post`, `teak tiller`: `rotation.y` = rudder angle.
- Hide the jib group when the headsail selected is not the jib (the genoa reuses the jib meshes scaled by 1.3 in x
  from its tack; gennaker, spinnaker and code 0 are new primitives, see below).

## Parts to ADD as primitives (named so the glossary can highlight them)

Children of the model group (model coordinates, y before the -1.5 shift), simple schematic geometry:
- `foil_port` and `foil_starboard`: C-foils. Each is a `THREE.Group` named `foil_port`/`foil_starboard` placed at the hull
  exit point (x = 1.0, y = 1.7, z = -1.25 / +1.25). Inside: `foil_shaft` = `THREE.TubeGeometry` (radius 0.07, 48 segments)
  along a `CatmullRomCurve3` going outboard and down then curving back inboard: local points
  (0,0,0), (0,-0.9,±0.75), (0,-1.9,±0.9), (0,-2.6,±0.45) (sign = side); `foil_tip` = `BoxGeometry(0.45, 0.06, 1.3)` centred at
  (0,-2.65,∓0.25) (goes inboard under the hull), slightly rotated so it looks like a wing. Material: dark grey, metallic.
  Cant = `group.rotation.x` (sign mirrored on the port side so positive cant moves both tips outboard/up), rake =
  `group.rotation.z` (positive = leading edge up), extension in [0,1] translates the foil contents along the initial
  shaft tangent so that 0 = raised 2.4 m (retracted), 1 = fully deployed. Only visual, no physics.
- `elevator`: `BoxGeometry(0.4, 0.04, 0.9)` at (-3.85, 0.33, 0) under the rudder, in the rudder group.
- `bowsprit`: `CylinderGeometry(0.04, 0.04, 1.6)` horizontal along x from x = 4.0 to 5.6 at y = 2.3.
- Headsails as `THREE.Mesh` of a `THREE.ShapeGeometry` triangle in a group pivoting about y at the tack (bowsprit tip
  (5.5, 2.3, 0) for gennaker/spinnaker/code 0): `gennaker` (tack (5.5,2.3), head (0.95,12.2), clew (0.5,3.5), orange,
  double-sided, opacity 0.9), `spinnaker` (clew further out (−1.5,4.0), red), `code_zero` (clew (1.5,3.0), white, flatter).
  Only one headsail visible at a time (`sail-config`); the mainsail is always visible.
- `waterline`: a thin `THREE.Line`/`TorusGeometry` or box outline at model y = 1.5 around the hull (loop of points
  following an ellipse x ∈ [-3.9, 4.05], z = ±1.3), white.
- Glossary markers: for glossary entries that have a `marker` [x, y, z], create a small sphere (radius 0.12, hidden by
  default) named after the entry id at that position, shown when the entry is highlighted.
- Arrows (`THREE.ArrowHelper`, in the scene, updated every frame): `arrow_heading` (green, from the bow along the boat
  x axis, length 4), `arrow_course` (orange, heading rotated by the leeway angle about y, length 4),
  `arrow_true_wind` (blue) and `arrow_apparent_wind` (cyan) at the mast top, pointing in the direction the wind blows TO,
  length = 0.25 * speed in kt (min 0.5).
- Sea: `PlaneGeometry(200, 200)` at y = 0, blue, opacity 0.6, plus a `GridHelper`; `AxesHelper(3)` at the origin.
  Lights: hemisphere + directional. Background dark blue-grey. Loading overlay with text until the GLB is parsed.

## Coordinate conventions (boat frame = scene frame when yaw = 0)

- `+x` forward, `+y` up, `+z` starboard. Angles in degrees in the UI, radians in the code.
- Hierarchy: `scene > boatRoot` (yaw about y, position y = ride height) `> attitudeGroup` (heel about x then trim about z)
  `> modelGroup` (translated y = -1.5, contains the GLB + added parts).
- Heel: positive = mast leans to starboard (`attitudeGroup.rotation.x = -heel` in radians, check visually that +z goes
  down). Trim: positive = bow up. Yaw: positive = bow turns to starboard (clockwise seen from above, i.e.
  `boatRoot.rotation.y = -yaw`). Leeway: positive = the boat drifts to starboard of its heading. Rudder: positive =
  trailing edge to starboard.
- True wind angle TWA in [-180, 180]: positive = wind comes from the starboard side (starboard tack), negative = from
  port. Sails and boom are set on the LEEWARD side: TWA > 0 -> boom to port (-z); TWA < 0 -> boom to starboard.
  Boom angle in degrees = clamp(5 + 75 * (|TWA| - 35) / 145, 5, 80) for |TWA| >= 35, else 5 (luffing is ignored).
- Cant: positive = tips move outboard/up. Rake: positive = leading edge up. Extension in [0, 1].
- Apparent wind (pure function, no DOM): true wind velocity in the boat frame `TW = TWS * (-cos TWA, 0, -sin TWA)`,
  boat velocity `V = (boatSpeed, 0, 0)`, `AW = TW - V`; `AWS = |AW|`, `AWA = atan2(-AW.z, -AW.x)` in degrees (same sign
  convention as TWA), `VMG = boatSpeed * cos(TWA)` (positive upwind). Speeds in knots. Show AWA, AWS, VMG rounded to 1
  decimal in `readout-awa`, `readout-aws`, `readout-vmg`, plus heading and course in `readout-heading`, `readout-course`.
- Leeway only rotates `arrow_course` (the boat itself does not translate).

## Controls (exact element ids, all `<input type="range">` except the select; a `<span id="value-...">` next to each)

| id | parameter (state key) | min | max | step | default | label |
|---|---|---|---|---|---|---|
| `slider-heel` | heel | -40 | 40 | 1 | -15 | Heel (gîte) |
| `slider-trim` | trim | -15 | 15 | 0.5 | 2 | Trim / pitch (assiette) |
| `slider-yaw` | yaw | -180 | 180 | 1 | 0 | Yaw / heading (lacet, cap) |
| `slider-leeway` | leeway | -15 | 15 | 0.5 | 4 | Leeway (dérive) |
| `slider-cant` | cant | -20 | 40 | 1 | 10 | Cant (foil) |
| `slider-rake` | rake | -5 | 10 | 0.25 | 2 | Rake (foil) |
| `slider-foil-extension` | foilExtension | 0 | 1 | 0.05 | 1 | Foil extension |
| `slider-ride-height` | rideHeight | 0 | 1.5 | 0.05 | 0.6 | Ride height (hauteur de vol) |
| `slider-rudder` | rudder | -30 | 30 | 1 | 0 | Rudder (safran) |
| `slider-twa` | twa | -180 | 180 | 1 | 45 | TWA (angle du vent réel) |
| `slider-tws` | tws | 0 | 40 | 1 | 15 | TWS (vitesse du vent réel) |
| `slider-boat-speed` | boatSpeed | 0 | 40 | 0.5 | 18 | Boat speed (vitesse bateau) |

`<select id="sail-config">` (state key `sailConfig`) with option values `main_jib`, `main_genoa`, `main_gennaker`,
`main_spinnaker`, `main_code_zero` (default `main_jib`), labels in the form "Grand-voile + foc (mainsail + jib)".
Camera preset buttons `cam-3quarter`, `cam-side`, `cam-top`, `cam-stern`, `cam-bow`; `btn-reset` restores all defaults.
Each slider label has a small "?" that shows the glossary definition of the matching term in `definition-box` on hover/click.

Expose `window.VIZ = { state, setParam(name, value), getParam(name), applyState(), highlight(termId), TERMS, computeApparentWind(twa, tws, boatSpeed) }`.
`setParam` updates the state, the control and the scene; `applyState()` re-applies the whole state to the scene (idempotent);
`computeApparentWind` returns `{awa, aws, vmg}` and MUST be pure (no DOM, no THREE). Define `computeApparentWind` and the
`window.VIZ` object at the top of the script, before any DOM/THREE code, and guard the start-up with
`if (typeof document !== 'undefined' && typeof THREE !== 'undefined') { init(); }` — the acceptance suite executes the
script in Node with only `window = globalThis` defined, so nothing outside `init()` may touch `document`, `THREE` or `requestAnimationFrame`.

## Glossary (given, copy it verbatim)

The file `glossary.json` in the workdir is the glossary, written by the project owner: copy its content VERBATIM (same
entries, same text, same order) into `<script id="glossary-data" type="application/json">...</script>` and parse it at
start-up (`JSON.parse(document.getElementById('glossary-data').textContent)`). Entry fields: `id`, `en`, `fr`, `group`
(`attitude`, `foil`, `sail`, `wind`, `hull`), `definition` (French), optional `control` (id of the slider/select that drives
it), optional `nodes` (list of substrings of mesh names, or of names of added parts/arrows, to highlight), optional
`marker` [x, y, z] (model coordinates, show the marker sphere). Do not edit the definitions.

Panel: scrollable column with the 5 groups titled "Attitude du bateau", "Foils et appendices", "Voiles et gréement",
"Vent et allures", "Coque", each term as "English (français)". Click = definition in `definition-box` (title + text) and
highlight of the matching meshes/markers for 2 s (emissive colour `0xffaa00` set on a cloned material, then restored).
`<input id="glossary-search">` filters the list (case- and accent-insensitive).

## Layout

Full-window dark theme: 3D canvas filling the left part (resize handler), a right column (width ~360 px) with the sliders
on top and the glossary below (both scroll), readout panel as an overlay on the canvas top-left (AWA, AWS, VMG, heading,
course), camera buttons overlay bottom-left. Usable at 1280x800. No console errors.

## build_standalone.py

`python build_standalone.py` reads `index.html` and `assets/sailing-yacht-8-m.glb`, and writes `sailing_viz_standalone.html`
where a `<script>window.MODEL_B64 = "<base64>";</script>` is inserted just before the application script. In `index.html`,
`loadModel()` uses `window.MODEL_B64` when defined (decode to an ArrayBuffer, `loader.parse(buffer, '', onLoad, onError)`)
and otherwise `loader.load('assets/sailing-yacht-8-m.glb', onLoad, undefined, onError)`.

## Deliverables (write nothing else; never modify `assets/` or `glossary.json`)

- `index.html`
- `build_standalone.py`
- `serve.bat` (one line: `python -m http.server 8765`)
- `tests/test_structure.py`: your own pytest tests (ids present, glossary block equals `glossary.json`, inline script passes
  `node --check`, `computeApparentWind` runs in Node). Node 18 is available as `node`.
- `README.md`: ~15 lines in French: how to open (serve.bat / standalone), what each slider does, model credit (Innerscene, CC0).

An independent acceptance suite is run on your files when you call `finish` (structure, glossary identity, JS syntax,
apparent-wind values against an independent reference, standalone build).
