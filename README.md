# Qwen-Projects — evaluating a local Qwen model as a programming agent

> English version. The French version is in [`README-fr.md`](README-fr.md).

This repository collects the trials and the study devoted to running a local
**Qwen3.8-27B** model (`qwen-science`, served by llama.cpp, 200k context, tools and images),
driven as an agent by a harness (`qwen_agent.py`) to write medium-sized Python code to completion:
package, tests, example, README.

The goal is not the delivered code itself, but **measuring what the model can do on its own**, and
what it needs to be given (tests, formulas, a verification method) for the deliverable to be *right*
— not merely "green". An independent judge (an acceptance test suite written by Claude, never shown
to the agent unless a prompt variant says so) arbitrates every pass.

## The evaluation project: `gp_lite`

Every pass re-uses the same task statement ([`gp_lite_v5/TASK.md`](gp_lite_v5/TASK.md)): write a small
self-contained Gaussian-process regression package (NumPy + SciPy only, no scikit-learn):

- stationary ARD kernels — SquaredExponential, Matérn 3/2, Matérn 5/2;
- `fit` by maximum marginal likelihood (L-BFGS-B, restarts, jitter);
- a posteriori prediction;
- **closed-form leave-one-out** residuals (Rasmussen & Williams, § 5.4.2);
- a pytest suite, an example on the Branin function with a plot, and a README.

The problem is deliberately sensitive to linear algebra (log-determinant, quadratic term, LOO) and to
optimisation bounds: it separates a package that *passes its own tests* from a package that is
*correct*. The read-only reference for the agent is the `Multifidelity-GP-Clean` repository.

## Layout

```
├── acceptance/                    # Claude's acceptance tests (23 tests) — the judge
│   └── test_acceptance.py
├── gp_lite/ … gp_lite_v5/         # the 5 successive passes of the first attempt (v1 to v5)
│   ├── gp_lite/                   #    each pass: package, tests, example, README, log
│   ├── gp_lite_v2/ … gp_lite_v5/
├── study/                         # prompt study: 4 variants (A-D) × 2 runs
│   ├── run_study.py               #    orchestration (serving, runs, scoring, results)
│   ├── make_plots.py              #    interactive figures
│   ├── prompts/                   #    base statement + addition of each variant
│   ├── runs/                      #    Qwen deliverables, logs (.qwen_agent/, agent_run.log)
│   ├── results/                   #    per-test scores + final report (JSON)
│   ├── fixes/                     #    Claude's fixes (diffs); the deliverables are not modified
│   ├── summaries/                 #    summaries of the 8 logs
│   └── RAPPORT_ETUDE.md           #    <— the outcome of the study
├── Bimodal-Sailing-Discretizer/   # one-off task: web editor for an operating profile
├── qwen_task_stats/               # one-off task: long-term statistics of Qwen tasks
├── RAPPORT_AGENT.md               # <— report of the first attempt (gp_lite, 5 passes)
├── README.md                      # this file (English)
└── README-fr.md                   # French version
```

## Two key moments

### 1. First attempt — [`RAPPORT_AGENT.md`](RAPPORT_AGENT.md) (26 October)

Five successive passes, each harness modified in reaction to the previous failures. The 5th pass
delivers a complete project (8/8 tests, honest report) — yet with a **mathematical bug** its own tests
cannot see: the quadratic term of the log-likelihood is `yᵀL⁻¹y` instead of `yᵀK⁻¹y`. The
`gp_lite/` through `gp_lite_v4/` folders are kept for reference.

> Takeaway: *a "green" project is not a correct project*. Its explanations of a failure may be invented
> ("environment bug").

### 2. Prompt study — [`study/RAPPORT_ETUDE.md`](study/RAPPORT_ETUDE.md) (27 October)

Four prompt variants, 2 runs each (8 runs, ~4 h of GPU time):

| Variant | Addition to the statement |
|---|---|
| A | the statement alone |
| B | + the acceptance tests provided |
| C | + the formulas and conventions given |
| D | + an imposed verification method |

**Result:** all 8 runs deliver a usable package, but **only 3 are right on the first try**; the other 5
become correct after 1 to 3 lines fixed by Claude, guided by the acceptance tests. No single variant is
reliable on its own. The most frequent defect is not a wrong formula: it is **optimisation bounds set too
narrow**, which the agent's own tests do not detect.

> Recipe kept: **Claude writes the acceptance tests, Qwen writes the code, Claude fixes it.** Qwen's runs
> cost only GPU time — so run two and keep the best one.

## Rerun

> Paths specific to this machine (Windows, `SIM`); adapt `qwen_agent.py`, the read directory and the
> llama.cpp server before running.

**First attempt**

```
cd C:\Users\SIM\Desktop\Code-Adri\Qwen-Projects\<folder>
python C:/Users/SIM/.claude/tools/qwen_agent.py @TASK.md --workdir . ^
  --read-root C:/Users/SIM/Desktop/Code-Adri --effort medium --max-steps 80 --max-minutes 90 ^
  --deliverables gp_lite/kernels.py,gp_lite/gp.py,tests/test_gp_lite.py,examples/branin_demo.py,README.md ^
  --test-cmd "-m pytest -q tests"
```

**Prompt study**

```
cd C:\Users\SIM\Desktop\Code-Adri\Qwen-Projects\study
python run_study.py            # reruns the runs missing from results/
python run_study.py C1 C2      # only these runs
python make_plots.py           # figures/etude_prompts.html
```

**The judge** (always on a pristine copy)

```
python -m pytest -q C:/Users/SIM/Desktop/Code-Adri/Qwen-Projects/acceptance/test_acceptance.py
# or, pointing GP_LITE_DIR at the deliverable to judge:
set GP_LITE_DIR=C:\...\Qwen-Projects\gp_lite_v5 && python -m pytest -q ..\acceptance\test_acceptance.py
```

## Reference files

- [`RAPPORT_AGENT.md`](RAPPORT_AGENT.md) — detailed report of the first attempt (passes, causes of the
  failures, independent verification, bugs).
- [`study/RAPPORT_ETUDE.md`](study/RAPPORT_ETUDE.md) — protocol, table of the 8 runs, what works / what
  does not, recommendations.
- [`gp_lite_v5/README.md`](gp_lite_v5/README.md) — documentation of the delivered package (the best pass).
- [`acceptance/test_acceptance.py`](acceptance/test_acceptance.py) — the 23-test judge suite.
- Each pass keeps its log: `<folder>/.qwen_agent/log.md` (thought, tool calls, outputs) and
  `agent_run.log`.

