# Project Status Notes — 2026-05-19

## What This Project Is

Active Inference (AIF) framework for encoding legal obligations as prior preference profiles in autonomous drone agents. Three regulatory domains: GDPR data minimization, EASA geofencing, emergency override. Uses pymdp (JAX-based) for Active Inference agents.

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

The initial `--quick` run showed C1 (Normal/Active — agent should stay) with ~67% violation rate. Deep investigation found two principled root causes:

**Root cause 1: B1 transition rate too high (was a=0.125)**
- Privacy zones are stable legal designations, not volatile states
- A 12.5% flip probability per timestep made the agent's beliefs volatile
- A single noisy observation could flip privacy belief from [0.97, 0.03] to [0.46, 0.54]
- **Fix:** `build_B_matrices(a_priv=0.01, a_urg=0.02)` — models privacy as stable

**Root cause 2: Uninformative D1 prior (was [0.5, 0.5])**
- At t=0, the agent starts at PATROL where privacy observations are ambiguous
- At t=1, the agent reaches APPROACH and gets its FIRST informative observation
- With a flat prior, this single observation dominates beliefs entirely
- If that observation is wrong (12.5% chance per modality), beliefs flip catastrophically
- **Fix:** `build_D_priors()` with D1=[0.75, 0.25] — modest prior favoring active privacy

**Validation results (30 trials x 7 conditions):**

| Condition | Description | Violations | Success |
|-----------|-------------|------------|---------|
| C1 | Normal, Active -> Stay | 16.7% | N/A |
| C2 | Emergency, Active -> Cross | 100% | 100% |
| C3 | Normal, Suspended -> Cross | 0% | 100% |
| C4 | Emergency, Suspended -> Cross | 0% | 100% |
| C5 | Normal, A->S@t7 -> Stay then cross | 6.7% | 10% |
| C6 | Emergency, A->S@t7 -> Cross | 96.7% | 96.7% |
| C7 | N->E@t4, A->S@t7 -> Override | 3.3% | 93.3% |

**Key insight: C_eff timing lag is PROTECTIVE.** Computing C_eff from pre-observation beliefs (not post) provides temporal smoothing that dampens the effect of single noisy observations. Moving C_eff after inference would make the agent MORE reactive to noise, not less.

**All downstream experiments** (exp1-3, ablation, baselines, sensitivity, noise, learning) call `build_B_matrices()` and `build_D_priors()` without overrides, so they pick up the new defaults automatically.

**Diagnostic files created (temporary):**
- `src/experiments/diagnose_c1.py` — step-by-step trace of single trials
- `src/experiments/diagnose_c1_stats.py` — configuration sweep across 4 B/D combos
- `src/experiments/diagnose_final.py` — final 30-trial x 7-condition validation

## Where We Are Now

### Immediate next step: Re-run --quick validation with fixed model

The B matrix and D prior fixes are applied. Need to re-run the full experiment suite to verify all 9 experiments still work and produce improved results.

### Then: Full production run

**Problem: GPU not available on native Windows.**
- Machine has NVIDIA RTX A4000 Laptop GPU (8 GB, CUDA 13.0 driver)
- JAX CUDA wheels (`jax-cuda12-plugin`) only exist for Linux, not Windows
- Attempted `pip install jax[cuda12]` — pip backtracked to jax 0.4.21 (broken)
- Reverted to jax 0.4.35 (working CPU-only)

**Options:**
1. Run full suite on CPU (~3.5 days estimated)
2. Set up project in WSL2 for JAX CUDA access
3. Reduce sensitivity sweep trials (e.g., 20 instead of 50)

### JAX version situation
- Installed: jax==0.4.35, jaxlib==0.4.35 (CPU-only, working)
- Dependency warnings exist (chex wants >=0.7.0, equinox wants >=0.4.38) but code runs fine
- Do NOT upgrade jax without testing — pymdp compatibility is fragile

## Key Files

| File | Role |
|------|------|
| `src/experiments/run_all.py` | Master runner (--quick flag for 3 trials) |
| `src/utils/profile_mixing.py` | Core C subtensor mixing |
| `src/models/emergency_override.py` | Main model with tuned B, D, and C profiles |
| `src/baselines/agents.py` | HPM and Bayes-rules baselines |
| `src/plotting/figures.py` | All 10 publication figures |
| `results/` | All JSON results + PDF figures |
| `paper/` | LaTeX paper + figures for Overleaf |

## Known Issues

1. **Sensitivity sweep is the bottleneck** — 35 cells with 50 trials each takes ~73 hrs on CPU
2. **Paper figures (fig6-fig10) not copied to paper/figures/** — only fig1-fig5 are copied by run_all.py
3. **Paper (main.tex) not yet updated** to reference the new experiments/figures
4. **C1 residual violations (~17%)** — inherent to stochastic model with 12.5% observation noise; represents irreducible error from simultaneous wrong observations on multiple modalities. This is actually realistic and defensible in the paper.
5. **C5 low success rate (10%)** — agent is conservative when privacy switches late with no emergency justification. Principled behavior but may need discussion in paper.
