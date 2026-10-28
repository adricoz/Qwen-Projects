The file `index.html` in this folder is CORRUPTED (a second <html>/<head>/<body> block was spliced into the
middle of the <style> of the old file). Do NOT patch it: REWRITE `index.html` ENTIRELY, as one complete file,
with your file write tool. Do NOT paste the code in the chat. At the end, list what you did in 5 lines max,
and only claim what is really in the file.

GOAL: the page is an editor of the operating profile used by our optimizer (Python class
`operating_profile.OperatingProfile`). Its export must be loadable as is by that class.

Libraries (only these): <script src="https://cdn.tailwindcss.com"></script> and
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>. Everything else inline, in ONE file.

0. LOOK (plain)
   White page, light Tailwind grays only (bg-white / bg-gray-50, border-gray-300, text-gray-900/600).
   Normal OS sliders (no custom slider CSS), simple bordered inputs, plain buttons. No dark theme, no glow,
   no icons. Layout: left sidebar (w-80, scrollable) with the controls, main panel with the plots.

1. MODEL
   Each mode m (1 = upwind, 2 = downwind) is a product of two independent TRUNCATED normal laws:
     p(TWA, TWS) = sum_m w_m * f(TWA; mu_twa, sd_twa, lo_twa, hi_twa) * f(TWS; mu_tws, sd_tws, lo_tws, hi_tws)
     f(x; mu, sd, lo, hi) = phi((x - mu)/sd) / (sd * (Phi(b) - Phi(a)))  if lo <= x <= hi, else 0,
     a = (lo - mu)/sd, b = (hi - mu)/sd, phi(z) = exp(-z^2/2)/sqrt(2 pi), Phi = standard normal CDF.
   w_2 = 1 - w_1. Do NOT renormalise over the grid: this density is already normalised.
   Sidebar cards: "Plotted domain" (N, TWA min/max, TWS min/max: display only), "Global weight" (w1),
   "Mode 1 · upwind" and "Mode 2 · downwind", each with: TWA mean, TWA sd, TWA min, TWA max, TWS mean,
   TWS sd, TWS min, TWS max. Each mode's bounds are clamped inside the plotted domain, and min < max.
   The surface shows p on the N x N grid of the plotted domain (N = mesh resolution, display only).

2. QUADRATURE NODES (what the optimizer really uses)
   Slider "Nodes per dimension" n (integer 2..6, default 3). For each mode, the TWA nodes are the n
   quantiles q_i = (i + 0.5)/n, i = 0..n-1, of the truncated law:
     x_i = mu + sd * PhiInv( Phi(a) + q_i * (Phi(b) - Phi(a)) )
   same for TWS; the mode's nodes are all n x n pairs, each with weight w_m / n^2. Total 2 n^2 nodes,
   weights summing to 1.
   - Phi(z) = 0.5 * erfc(-z / sqrt(2)); implement erfc with the Numerical Recipes Chebyshev formula
     (erfccheb) or W. J. Cody's rational approximations (absolute error < 1e-12), NOT Abramowitz-Stegun 7.1.26.
   - PhiInv: Peter Acklam's rational approximation followed by ONE Halley refinement step using Phi.
   - If Phi(b) - Phi(a) < 1e-12 (mode entirely outside its bounds), show a red warning in that mode's card
     and disable the export.
   Show the nodes as markers ON the 3D surface (z = p at the node), size proportional to the weight,
   hover "mode, TWA, TWS, weight". Text line above the plots: "2 n^2 = K nodes · peak p = ...".
   Under the 3D plot (main panel split 60/40): a 2D top view, Plotly heatmap of p (same colorscale,
   selector Viridis / Jet / Rainbow), nodes overlaid (blue circles = upwind, orange = downwind) and the
   rectangle of each mode's bounds.

3. CONTROLS
   - Every slider has a small text input next to it (w-16, right-aligned) showing its value. Typing a value
     and pressing Enter or leaving the field sets the slider: parse the number, snap to the slider step,
     clamp to [min, max]; invalid text restores the current value. Moving the slider updates the field.
   - Camera persistence: before each Plotly.react of the 3D plot, read the current camera from
     plotDiv._fullLayout.scene.camera and put it in the new layout (layout.scene.camera), and set
     layout.uirevision = 'keep' on both plots, so moving a slider never resets the view or the zoom.
     The axis ranges of both plots are recomputed only when the plotted domain bounds change.
   - Button "Reset view" (top-right of the main panel): default camera eye (1.55, -1.55, 0.72) and ranges
     fitted to the plotted domain. Button "Reset to defaults" in the sidebar: all values of section 6.

4. EXPORT
   Text input "Profile name" (default "bimodal_profile", allowed chars [A-Za-z0-9_-], others removed) and a
   textarea "Comment". Button "Export profile JSON" writes `<name>.json`, 1-space indented
   (JSON.stringify(obj, null, 1)), EXACTLY this structure (numbers rounded to 2 decimals, weights to 4):
   {
    "name": "<name>",
    "comment": "<comment>",
    "nodes_per_dim": 3,
    "modes": [
     {"name": "upwind", "weight": 0.5,
      "twa": {"mean": 45.0, "std": 4.0, "lower": 38.0, "upper": 52.0},
      "tws": {"mean": 13.0, "std": 3.0, "lower": 8.0, "upper": 18.0}},
     {"name": "downwind", "weight": 0.5,
      "twa": {"mean": 135.0, "std": 7.0, "lower": 120.0, "upper": 148.0},
      "tws": {"mean": 13.0, "std": 3.0, "lower": 8.0, "upper": 18.0}}
    ]
   }
   No other keys (no nodes, no grid, no flat array).
   Saving: if window.showSaveFilePicker exists, open it (suggestedName `<name>.json`, type application/json)
   and write the file; if the user cancels (AbortError), do nothing. Otherwise fall back to a normal
   download (Blob + <a download>).

5. IMPORT
   Button "Load profile JSON" (hidden <input type="file" accept=".json">): reads a file of the structure
   above (exactly 2 modes, the first is upwind), sets all sliders, text fields, name, comment and n, then
   redraws. Values outside a slider's range: clamp and show a small notice. Invalid file: show the error,
   change nothing.

6. DEFAULTS (= the reference profile above): w1 0.5; upwind TWA 45 sd 4 [38, 52], TWS 13 sd 3 [8, 18];
   downwind TWA 135 sd 7 [120, 148], TWS 13 sd 3 [8, 18]; n 3; plotted domain TWA [25, 165], TWS [4, 36];
   N 40. Slider ranges: N 10..80; domain TWA 0..180, TWS 0..40; means TWA 0..180 step 1, TWS 0..40 step 0.5;
   sd TWA 1..45 step 0.5, sd TWS 0.5..15 step 0.5; bounds step 1 deg / 0.5 kt; w1 0..1 step 0.01.

7. SELF-CHECK (required): a function `selfTest()` run at load compares the nodes of the defaults with these
   reference values (scipy.stats.truncnorm) and logs PASS/FAIL to the console, and shows a red banner on FAIL:
     upwind   TWA nodes 41.5379, 45.0000, 48.4621 ; TWS nodes 10.4593, 13.0000, 15.5407
     downwind TWA nodes 128.4518, 134.8633, 141.1360 ; TWS nodes 10.4593, 13.0000, 15.5407
     extra case: mean 140, sd 25, [25, 165], n = 4 -> 108.6842, 127.9924, 141.6205, 155.7901
   Tolerance 1e-3. Also check that the 2 n^2 weights sum to 1 within 1e-12.
