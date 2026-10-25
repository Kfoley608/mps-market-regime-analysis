# The four gauges and their classical cousins, decoded

These notes explain the measures used in the Week 6 analysis. The later [out-of-sample validation](../week7/FINDINGS.md) found no consistent forecasting advantage, so the gauges should be read as descriptions of dependence rather than trading signals.

A plain-language companion to the "§1 — four gauges" slide: what every MPS column is, how it is
calculated, what its classical cousin is, and what each pairing means.

## The shared machinery (applies to every MPS gauge)

All four MPS gauges come from the same rolling battery: a **250-trading-day window** slides through
2009–2026 in 10-day steps (434 windows). In each window, every asset's returns are cut into **8
quantile bins** (bin 0 = that asset's worst 12.5% of days in the window, bin 7 = its best), and an
**MPS Born machine with bond dimension χ=8** is trained by DMRG on the binned panel. The fitted
object is a full joint probability distribution over the 12-asset day; each gauge is one number
read off that distribution. Everything is causal: fits use only the trailing window.

---

## Row 1 — Complexity / Fragility (`G_complexity_HO`) vs `absorption_ratio`

**The MPS gauge.** Four models are fitted per window and scored on walk-forward predictive
log-likelihood (how much probability they assigned to the days after the window):

| Model | Sees |
|---|---|
| Born χ8 | any dependence shape |
| Born χ1 | nothing — independence (a χ=1 bond carries zero information between assets) |
| Gaussian, full covariance | linear dependence only |
| Gaussian, diagonal covariance | nothing — independent Gaussians |

The gauge is **(Born χ8 − Born χ1) − (Gauss full − Gauss diag)**, 21-day smoothed. Bracket 1 is
the predictive value of *all* dependence; bracket 2 is the value of *correlation alone*; the
difference is the value of dependence a covariance matrix cannot express. Subtracting only
*within* each family keeps discrete-vs-continuous scoring comparable — the bin-width terms in the
two Born scores cancel exactly (the "same-measure" fix, BUG-LOG #18a: the original version
subtracted a continuous Gaussian quantity from a discrete Born one and quietly mixed
discretization error into the answer). Reading: ≈0 → covariance describes the joint behavior;
above 0 → predictive structure beyond covariance; below 0 happens too — the four legs are
separately fitted models, so this is a predictive-score contrast, not a literal total-correlation
estimate.

**The cousin: absorption ratio** (Kritzman–Li–Page–Rigobon). Eigendecompose the window's
correlation matrix and take the **share of total variance absorbed by the top n/5 eigenvectors**
(for 12 assets: the top 2–3). High absorption = the market is dancing to a few common factors =
tightly coupled = fragile. It is the industry-standard fragility gauge — and it is built entirely
from the covariance matrix, so it can only see *linear* coupling.

**The pairing means:** both ask "how coupled is the market?" The cousin answers with covariance;
the MPS version asks whether there is coupling *beyond* what covariance can say. They track each
other in broad strokes (~50% shared variance) — the honest framing is corroboration plus a
higher-order layer, not a novel signal.

---

## Row 2 — Joint-Crash (`G_jointcrash`) vs `turbulence`

**The MPS gauge** (feature F14, `kway_corner_probability`, k=3, q=5%). For every one of the
C(12,3) = 220 possible triples of assets, contract the fitted MPS **exactly** (no sampling) to get
the joint probability that all three sit in their bottom tail-bin on the same day; the gauge is
the mean over all triples. The tail set per asset is the bottom m bins with m = max(1, round(q·d))
— with d=8 and q=5% that is the single bottom bin, ≈12.5% of each asset's own days. The
independence baseline is (12.5%)³ ≈ 0.2%, so readings far above that are pure dependence. In plain
English: **the model's estimate of the chance that a cluster of assets crashes together today.**

**The cousin: turbulence** (Kritzman's Mahalanobis measure). For each day,
d_t = (r_t − μ)′ Σ⁻¹ (r_t − μ) — how statistically unusual today's return vector is relative to a
trailing mean μ and covariance Σ. Big moves in strange directions score high.

**The pairing means:** both are "something is breaking" meters. Turbulence measures how unusual
today looks *given the covariance geometry* — it inherits covariance's assumptions. The Joint-Crash
gauge measures the probability mass the fitted joint distribution puts on the specific event risk
desks fear: simultaneous multi-asset crashes. About 25% of its variation is new relative to the
whole classical panel — a supporting instrument, not the headline.

---

## Row 3 — Fear vs Euphoria (`G_asymmetry`) vs `realized_vol`

**The MPS gauge** (feature F12, tail asymmetry). Same corner mathematics as Joint-Crash, run on
**both ends** of the distribution: joint **lower**-corner mass (co-crash) minus joint
**upper**-corner mass (co-rally), read off the fitted MPS. Positive = assets glue together in
selloffs more than in rallies (fear); negative = coordinated upside dominates (euphoria). This is
the co-skewness direction of the joint distribution — and it is the headline gauge because it is
**structurally invisible to any symmetric model**: a Gaussian's co-skew is identically zero, no
matter what covariance you give it. Empirically ~88–95% of its variance is unexplained by the full
classical panel, on both books tested.

**The cousin: realized volatility.** The average of the 12 assets' return standard deviations over
the window, annualized (×√252). The most basic risk number there is.

**The pairing means:** realized vol measures the *size* of moves and is completely
direction-blind — a window of coordinated melt-ups and a window of coordinated crashes can have
identical vol. The asymmetry gauge is precisely the direction information vol throws away. That is
why this row is the cleanest "beyond covariance" demonstration in the module.

---

## Row 4 — Regime / Early-Warning (`p_stress`) vs `VIX`

**The MPS gauge.** A **Markov-switching Born machine**: two regimes (calm and stress), each with
its own MPS emission model, linked by a sticky transition matrix, estimated on the full panel and
then run as a **causal filter** — p_stress(t) = P(stress regime | data up to day t). It is
near-binary by design (sticky transitions make the filter commit to a regime). Important Week-7
caveat, always attached: this is a *descriptive regime label*, not a calibrated probability of a
future crash — walk-forward testing showed that using it as a forecast is a category error.

**The cousin: VIX.** The CBOE's option-implied 30-day S&P 500 volatility — the market's *priced*
expectation of near-term turbulence, i.e., what fear costs in the options market.

**The pairing means:** both try to answer "what state are we in right now?" VIX answers with the
options market's aggregate opinion about future *volatility level*; p_stress answers with which
*dependence structure* currently fits the data better. They disagree often (~90% of p_stress's
variance is unexplained by the classical panel) — which is exactly what makes it a second, largely
independent regime opinion rather than an echo of VIX.

---

## The other classical panel members (used in the redundancy bar)

The orthogonality regression tests each gauge against **all seven** classical indicators at once,
not just its named cousin. The other three:

- **Average correlation** — mean off-diagonal pairwise correlation of the window. The simplest
  coupling number.
- **PCA-1 share** — fraction of variance explained by the first principal component of the
  correlation matrix; close kin of the absorption ratio (which uses the top n/5 components).
- **MOVE** — the bond-market analog of VIX: option-implied Treasury volatility.

A gauge only counts as "new" if the *combination* of all seven cannot reproduce it — that is the
redundancy bar every number on the orthogonality slide is measured against.

## One-sentence versions for the presentation

- **Complexity/Fragility:** "How much does knowing the co-movement help predict tomorrow, beyond
  what correlation alone would help? — measured as a model-score gap."
- **Joint-Crash:** "The fitted distribution's probability that three or more assets have a
  bottom-octile day simultaneously — computed exactly, averaged over all 220 triples."
- **Fear vs Euphoria:** "Joint-crash mass minus joint-rally mass — the direction of togetherness,
  which every symmetric model scores as exactly zero."
- **Regime:** "Which dependence regime — calm or stress — currently explains the data better,
  filtered causally; a label, not a forecast."
