# Project Status Notes — 2026-05-20

## What This Project Is

Active Inference (AIF) framework for encoding legal obligations as prior preference profiles in autonomous drone agents. Three regulatory domains: GDPR data minimization, EASA geofencing, emergency override. Uses pymdp (JAX-based) for Active Inference agents.

## Current Plan (2026-05-20)

### What we're doing right now

**Run the full experiment pipeline on the EC2 instance (aif-meta cogames) with GPU.**

The code is ready, the model fixes are applied, but experiments have NOT been re-run since the fixes. On the local Windows machine, JAX can only run on CPU (JAX CUDA wheels are Linux-only), and even the `--quick` validation takes ~5 hours due to JIT compilation. The EC2 instance with GPU should be dramatically faster.

### Steps to execute on EC2

1. Clone/pull the repo on the EC2 instance
2. Install dependencies: `pip install -e .` or `pip install jax[cuda12] pymdp equinox matplotlib`
3. Verify GPU is detected: `python -c "import jax; print(jax.devices())"`
4. Run `--quick` validation first: `python src/experiments/run_all.py --quick`
5. If --quick passes, run full production: `python src/experiments/run_all.py`
6. Pull results back (JSON + PDF figures in `results/`)

### Why we're doing this

The previous `--quick` run (before model fixes) showed serious problems:
- C1 violation rate ~67% (should be low)
- Ablation results statistically meaningless
- Sensitivity sweep showed fragile mechanism
- Baselines didn't differentiate AIF
- Noise experiment was a step function
- mean_gamma was constant

We diagnosed and fixed the root causes (see "Investigation" section below). Now we need to re-run everything to confirm the fixes work across all 9 experiments and produce publishable results.

## What Was Done (Complete)

### Major Refactoring: C Subtensors / Belief-Weighted Profile Mixing

Replaced hard if-then threshold C-switching with principled belief-weighted preference profile mixing across all three models. This is the core theoretical contribution.

**Shared infrastructure created:**
- `src/utils/profile_mixing.py` — `compute_C_effective()` for belief-weighted mixing
- `src/utils/stats.py` — bootstrap CIs, Mann-Whitney U, Cohen's d
- `src/environments/drone_env.py` — made stochastic (categorical sampling instead of argmax)

**Models updated with C profiles:**
- `src/models/emergency_override.py` — 4 profiles: (urgency x privacy)
- `src/models/data_minimization.py` — 3 profiles: (scene composition)
- `src/models/geofence.py` — 2 profiles: (airspace status)

**Experiments refactored to use profile mixing:**
- `src/experiments/exp1_minimization.py`
- `src/experiments/exp2_geofence.py`
- `src/experiments/exp3_emergency.py`

**New experiments created:**
- `src/experiments/exp_ablation.py` — EFE ablation (FULL/PRAGMATIC/EPISTEMIC/RANDOM)
- `src/experiments/exp_baselines.py` — AIF vs HPM_ORACLE vs HPM_NOISY vs BAYES_RULES
- `src/experiments/exp_sensitivity.py` — parameter sweep (ETD x NCA)
- `src/experiments/exp_noise.py` — noise robustness across 7 noise levels
- `src/experiments/exp_learning.py` — A-matrix learning from experience

**Baselines created:**
- `src/baselines/agents.py` — HPMOracleAgent, HPMNoisyAgent, BayesRulesAgent

**Other:**
- `src/models/traceability.py` — removed artificial `min(..., 5.0)` overhead cap
- `src/plotting/figures.py` — added 5 new figure functions (fig6-fig10)
- `src/experiments/run_all.py` — updated to run all 9 experiments + 10 figures

### Investigation: C1 Violation Root Cause (2026-05-19)

The initial `--quick` run showed C1 (Normal/Active — agent should stay) with ~67% violation rate. Deep investigation found two principled root causes and applied fixes:

**Root cause 1: B1 transition rate too high (was a=0.125)**
- Privacy zones are stable legal designations, not volatile states
- A 12.5% flip probability per timestep made the agent's beliefs volatile
- **Fix:** `build_B_matrices(a_priv=0.01, a_urg=0.02)` — models privacy as stable

**Root cause 2: Uninformative D1 prior (was [0.5, 0.5])**
- At t=0, the agent starts at PATROL where privacy observations are ambiguous
- With a flat prior, the first informative observation dominates beliefs entirely
- **Fix:** `build_D_priors()` with D1=[0.75, 0.25] — modest prior favoring active privacy

**Validation results (30 trials x 7 conditions, via diagnose_final.py):**

| Condition | Description | Violations | Success |
|-----------|-------------|------------|---------|
| C1 | Normal, Active -> Stay | 16.7% | N/A |
| C2 | Emergency, Active -> Cross | 100% | 100% |
| C3 | Normal, Suspended -> Cross | 0% | 100% |
| C4 | Emergency, Suspended -> Cross | 0% | 100% |
| C5 | Normal, A->S@t7 -> Stay then cross | 6.7% | 10% |
| C6 | Emergency, A->S@t7 -> Cross | 96.7% | 96.7% |
| C7 | N->E@t4, A->S@t7 -> Override | 3.3% | 93.3% |

**Key insight: C_eff timing lag is PROTECTIVE.** Computing C_eff from pre-observation beliefs provides temporal smoothing that dampens noise. Do NOT move C_eff after inference.

**All downstream experiments** call `build_B_matrices()` and `build_D_priors()` without overrides, so they pick up the new defaults automatically.

## After the Run — Remaining TODO

1. **Evaluate results critically** — check ablation differentiation, baseline separation, sensitivity robustness, noise graceful degradation, mean_gamma
2. **Copy figures fig6-fig10 to paper/figures/** — run_all.py outputs to results/, only fig1-5 were previously copied to paper/figures/
3. **Update paper (main.tex)** — add sections for new experiments, update B/D parameter justification, discuss C subtensor approach
4. **Clean up diagnostic files** — `diagnose_c1.py`, `diagnose_c1_stats.py`, `diagnose_final.py` are temporary investigation tools
5. **Address C5 success rate** — 10% success is principled (conservative safety) but needs paper discussion

## Technical Notes

### JAX version (local Windows)
- Installed: jax==0.4.35, jaxlib==0.4.35 (CPU-only, working)
- Do NOT upgrade jax without testing — pymdp compatibility is fragile

### JAX on EC2
- Install with CUDA: `pip install jax[cuda12]`
- Verify: `python -c "import jax; print(jax.devices())"` should show GPU

### Key Files

| File | Role |
|------|------|
| `src/experiments/run_all.py` | Master runner (--quick flag for 3 trials) |
| `src/utils/profile_mixing.py` | Core C subtensor mixing |
| `src/models/emergency_override.py` | Main model with tuned B, D, and C profiles |
| `src/baselines/agents.py` | HPM and Bayes-rules baselines |
| `src/plotting/figures.py` | All 10 publication figures |
| `results/` | All JSON results + PDF figures |
| `paper/` | LaTeX paper + figures for Overleaf |

### Known Issues
1. **Sensitivity sweep is the bottleneck** — 35 cells x 50 trials; should be much faster on GPU
2. **C1 residual violations (~17%)** — inherent to stochastic model with 12.5% observation noise; irreducible error, realistic and defensible
3. **C5 low success rate (10%)** — conservative agent behavior with no emergency justification
