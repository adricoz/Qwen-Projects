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

