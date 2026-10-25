# Training Methodology — Why Adam Under-Fit Correlation, and the DMRG Fix

This note records an early training correction. The later [capacity and forecasting tests](../week7/FINDINGS.md) are the final reference for model performance.

**Quantum-Inspired Market Regime & Tail-Complexity Module · MFE Industry Project**

A major finding from this week's deep dive: our gradient-trained (Adam) MPS Born machine **under-fits
cross-asset correlation**, and a **proper canonical-form DMRG trainer fixes it**. This document records
the problem, the diagnosis, the evidence, the fix, and the implications — including an honest correction
to earlier claims.

---

## 1. The problem (and an honest correction)

Earlier I reported that the MPS "reproduces the correlation block structure (damped at χ=8)." That was
**wrong** — I mis-read a large correlation error as "damped." On the multi-asset books the MPS off-
diagonal correlation was **≈0.02–0.13 vs an empirical ~0.35**, i.e. **correlation was essentially not
captured.** What *was* genuinely verified and still holds: **volatility / marginals** (vol matches
realized to ~0.3–1%) and the small-N case (2 assets: 0.81 of 0.86). The correlation claim for N > ~4
should not have been made without checking magnitudes.

## 2. The symptom: correlation capture collapses with universe size N

Adam (d=8, χ=8), fraction of *achievable* (bin-mean) correlation captured:

| N | 2 | 4 | 8 | 12 | 24 |
|---|---|---|---|---|---|
| **Adam capture** | 94% | 65% | 37% | 25% | ~25% |

Marginals (vol) fit fine at every N; only the cross-asset **correlation** degrades, monotonically with N.

## 3. The diagnosis (mechanism, with evidence)

Three overlapping, *documented* causes — confirmed by experiment and literature:

- **Exponential correlation decay with chain distance (the smoking gun).** In a trained model, capture
  by chain distance |i−j|: **79% adjacent → 35% (dist 3) → ~1% (dist 8+).** 1-D MPS correlations decay
  with distance; the optimizer builds adjacent structure but fails to propagate global structure.
- **The "causality trap"** (Tang, Khoo & Ying, arXiv:2505.06419, 2025): randomly-initialized + gradient-
  trained MPS collapse into "causal models that ignore non-local correlations" — fit marginals, miss
  cross-variable structure, **worsening with N**. Over-parameterization does not escape it. *This is our
  symptom, named in the literature.*
- **Global-loss barren plateaus** (Liu et al., PRL 129 / arXiv:2108.08312): gradient variance vanishes
  **exponentially in N** for a global loss (our global NLL), but not for a local loss. The N-monotonic
  decay is the signature.

**Compounded by a structural reality:** equity correlation is dominated by a single long-range **market
factor** (PC1 ≈ 42% of variance; residuals ≈ uncorrelated), which a 1-D chain represents poorly.

**Ruled out, with evidence:** a math/contraction bug (verified correct; exact ΣP=1; vol matches; 2-asset
works); a read-off bug (exact `mps_correlation` and bin-mean *sampled* correlation agree); the
initialization scale (random ≈ near-identity — near-identity is the field-standard init, *not* the bug);
bond dimension as the *primary* cause (χ=8→32 barely moved the gap); and pure under-training (4000 steps
plateaus at ~0.13; lr/clip tuning → only ~50%).

## 4. The fix: canonical-form two-site DMRG

A proper DMRG/ALS trainer (`src/mps_regime/dmrg.py`) — mixed canonical form (so locally Z = ‖T‖², well-
conditioned), per-sample environment caching, local two-site updates + SVD truncation — **captures the
correlation Adam misses:**

| | achievable | Adam | **DMRG** |
|---|---|---|---|
| N=12 | 0.336 | 0.02 (6%) | **0.226 (67%)** |
| N=24 | 0.311 | 0.04 (13%) | **0.177–0.185 (57–59%)** |

A **5–10× improvement**, with lower NLL, exact normalization (≈1e-15; brute-force ΣP=1 confirmed), and
vol still matching. Stable across seeds. *Why it works where our quick "block-Adam + SVD" hack didn't:*
canonical form (local, well-conditioned objective — avoids the barren plateau) plus a critical numerical
fix (per-sample environment vectors underflow over long chains → row-normalize them).

## 5. Why Adam works for others but not us

Adam *does* train MPS Born machines successfully in the literature (Miller et al. 2020; Glasser et al.;
TorchMPS). The difference is **data + recipe**: those successes are on **strong, often local/sequential**
correlations (images, grammars), frequently **with canonical form / lr=1e-4 / many epochs**. Our data —
**weak (~0.3), long-range (market-factor), noisy** financial correlation — is exactly the regime where
the causality trap + barren plateau bite hardest. So **for our problem, global Adam is the wrong tool**,
and **DMRG/ALS is necessary** — which independently re-derives why the sponsor's FabriQ engine uses ALS.

## 5b. Ordering matters far more once the model actually works

Re-running the ordering benchmark with the **DMRG** trainer (the earlier benchmark used the broken Adam
model, so its ranking was unreliable). Correlation capture (% of achievable):

| Ordering | N=12 | N=24 |
|---|---|---|
| **Fiedler (MI)** | **87%** | **57%** |
| OLO (corr) | 71% | 54% |
| by-sector | 66% | 39% |
| alphabetical | 65% | 44% |
| random | 58–60% | 31–33% |

- **Ordering is now a first-order lever** — Fiedler 87% vs random 58% at N=12 (a 29-pt gap), where on the
  broken Adam model orderings differed only marginally. DMRG captures *adjacent* correlations best, so
  putting dependent assets adjacent (the MinLA objective) matters enormously.
- **Fiedler is decisively best** at both N, re-validated on a model that genuinely captures dependence.
- **DMRG + Fiedler captures most of the correlation (87% at N=12)** — largely solving the correlation
  problem with the right trainer + ordering; the N=24 gap (57%) is the χ-capacity limit.

## 5c. Beyond MPS+DMRG — alternative models (from the literature)

The residual N=24 gap is partly a **wrong-topology** issue: a 1-D chain has *exponentially*-decaying
correlations along the chain — a poor fit for a pervasive market factor. Higher-value options:

- **Learned-topology Tree Tensor Network (TTN).** Tree TNs have *algebraic* (not exponential) correlation
  decay. Harada, Okubo, Kawashima (arXiv:2408.10669) trained a learned-topology tree Born machine on the
  **S&P 500 and it autonomously recovered the 11 GICS sectors** — direct evidence a learned tree captures
  the equity factor/sector structure a 1-D MPS can't. Keep DMRG sweeps; add **TTNS-Sketch warm-start**
  (Tang-Khoo-Ying's documented causality-trap fix; ~128 samples sufficed).
- **Regime-switching vine copula** (Fink et al., arXiv:1604.05598) — the mature, interpretable workhorse
  for the regime/tail goal: per-day crisis-regime posterior + per-regime tail dependence, validated on
  equities. Strong candidate as the *primary* tool, with the TN entropy as a novel complement.
- **Honest flag:** training an MPS/DMRG on returns and using its **bond/entanglement entropy as a regime
  signal appears genuinely novel** (no prior empirical validation) — the research opportunity *and* the
  risk.

## 6. Implications

- **Re-base the analyses on DMRG.** Tier-0 (correlation/tail matrices), the showcase, Tier-1, and the
  higher-order-vs-Gaussian comparison must be redone on a model that actually captures dependence. Prior
  N>4 correlation/entanglement/Tier-1 results are unreliable and are being corrected.
- **The remaining gap (57–67%, not 100%) is now a genuine bond-dimension (χ) capacity limit** (χ=16
  lifted N=12 from 0.22→0.25) — so the χ-sweep is meaningful *for DMRG*.
- **The MPS's role, reframed:** linear equity correlation *is* the market factor (a 1-factor model gets
  it). The MPS's value, and the project's goal, is the **higher-order / tail / regime structure beyond
  the market factor** — "what standard methods miss."

## 7. References
- Tang, Khoo & Ying, *Initialization and training of MPS probabilistic models*, arXiv:2505.06419 (causality trap).
- Liu, Yu, Duan, Deng, PRL 129 270501 / arXiv:2108.08312 (global-loss barren plateaus).
- Miao & Barthel, PRA 109 L050402 / arXiv:2304.14320 (canonical form → plateau-free).
- Han, Wang, Fan, Wang, Zhang, PRX 8 031012 / arXiv:1709.01662 (DMRG-2 Born machine).
- Miller, Rabusseau, Terilla, arXiv:2003.01039 (Adam-trained generative MPS works on favorable data).
- Harada, Okubo, Kawashima, arXiv:2408.10669 (learned-topology tree TN recovers S&P 500 sectors).
- Cheng, Wang, Xiang, Zhang, PRB 99 155131 / arXiv:1901.02217 (tree TN generative; algebraic decay).
- Fink, Klimova, Czado, Stöber, arXiv:1604.05598 (regime-switching vine copulas for equity tail regimes).
