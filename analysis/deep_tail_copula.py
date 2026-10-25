"""Deep-tail k-way co-crash, CORRECTED null: Gaussian copula + empirical marginals.

Fixes vs deep_tail_sweep/book:
  1. Null is a Gaussian COPULA with the data's own (fat) marginals — so "empirical > null"
     can only come from dependence structure beyond a Gaussian copula, not from fat marginals.
     (The old continuous-Gaussian null is kept as a column to quantify how much of the old
     headline was a marginal artifact.)
  2. Tail sets defined by EMPIRICAL mass (deterministic bins), actual set mass printed —
     no mislabeled "1%" rows; at d=40 the deepest MPS-resolvable tail is 2.5%, but the
     empirical-vs-copula comparison is also run at the exact 1% quantile (MPS column blank).
  3. MPS tail-set marginal calibration printed (model tail mass vs empirical tail mass).
  4. Stationary block bootstrap CIs (Politis-Romano) on the empirical joint probability and
     on the emp/copula ratio; thresholds re-estimated inside each replicate.

Usage:
  python deep_tail_copula.py --synthetic            # validation: t5 marginals + Gaussian copula
  python deep_tail_copula.py <prices.parquet> [n_assets] [chi]
"""
import warnings; warnings.filterwarnings("ignore")
import sys, itertools
import numpy as np, pandas as pd
from scipy.stats import norm

sys.path.insert(0, "src")

B_BOOT, BLOCK_L, N_MC = 400, 20, 400_000
N_MC_BOOT = 100_000   # per-replicate null MC (BUG-LOG #18h); point estimate still uses N_MC
QS = (0.01, 0.025, 0.05, 0.10)
D_BINS = 40


def normal_scores_corr(R):
    T = R.shape[0]
    z = norm.ppf((np.argsort(np.argsort(R, axis=0), axis=0) + 0.5) / T)
    return np.corrcoef(z.T)


def kway_joint(lo, k, N):
    """Mean over all k-subsets of P(all below threshold). lo: (T,N) bool."""
    return float(np.mean([lo[:, list(s)].all(1).mean()
                          for s in itertools.combinations(range(N), k)]))


def copula_joint(Zsim, levels, k, N):
    """Gaussian-copula null: P(all k in tail) with per-asset marginal tail mass `levels`."""
    lo = Zsim < norm.ppf(levels)[None, :]
    return kway_joint(lo, k, N)


def stationary_bootstrap_idx(T, rng, expected_block=BLOCK_L):
    p = 1.0 / expected_block
    idx = np.empty(T, dtype=int)
    idx[0] = rng.integers(T)
    restart = rng.random(T) < p
    steps = rng.integers(T, size=T)
    for t in range(1, T):
        idx[t] = steps[t] if restart[t] else (idx[t - 1] + 1) % T
    return idx


def analyze(R, name, mps_fit=None, edges=None, p1=None):
    T, N = R.shape
    rng = np.random.default_rng(0)
    Rho = normal_scores_corr(R)
    Zsim = rng.standard_normal((N_MC, N)) @ np.linalg.cholesky(Rho).T
    Gold = rng.multivariate_normal(R.mean(0), np.cov(R.T), size=N_MC)  # the OLD (wrong) null

    print(f"\n===== {name} | T={T} N={N} | avg corr {np.corrcoef(R.T)[np.triu_indices(N,1)].mean():.2f} "
          f"| avg normal-scores corr {Rho[np.triu_indices(N,1)].mean():.2f} =====")
    hdr = (f"{'q':>6} {'k':>2} | {'set-mass':>8} | {'empirical':>10} {'[95% CI]':>19} | "
           f"{'copulaNull':>10} {'oldGauss':>9} {'indep':>9} {'MPS':>9} | "
           f"{'emp/cop':>8} {'[95% CI]':>13} | verdict vs COPULA null")
    print(hdr); print("-" * len(hdr))

    results = []
    for q in QS:
        # --- thresholds: on the MPS bin grid where possible, exact quantile otherwise
        if edges is not None and q >= 1.0 / D_BINS:
            m = int(round(q * D_BINS)) - 1
            thr = np.array([edges[j][m] for j in range(N)])
        else:
            m = None
            thr = np.quantile(R, q, axis=0)
        lo = R < thr
        mass = lo.mean(0)                      # actual empirical per-asset tail mass
        for k in (2, 3):
            emp = kway_joint(lo, k, N)
            cop = copula_joint(Zsim, mass, k, N)
            gld = kway_joint(Gold < thr, k, N)
            ind = float(np.mean([np.prod(mass[list(s)])
                                 for s in itertools.combinations(range(N), k)]))
            # --- MPS leg (analytic, bins 0..m) + calibration
            mps_p, calib = float("nan"), float("nan")
            if mps_fit is not None and m is not None:
                sets = [np.arange(0, m + 1)] * N
                mps_p = kway_mps(mps_fit, sets, k, N)
                calib = float(np.mean(np.abs(p1[:, :m + 1].sum(1) - mass) / mass))
            # --- stationary block bootstrap on emp (thresholds re-estimated per replicate)
            brat, bemp = [], []
            brng = np.random.default_rng(1)
            for _ in range(B_BOOT):
                Rb = R[stationary_bootstrap_idx(T, brng)]
                thr_b = np.quantile(Rb, mass.mean(), axis=0)
                lob = Rb < thr_b
                eb = kway_joint(lob, k, N)
                # REVIEW FIX 2026-07-14 (BUG-LOG #18h): re-estimate the copula null INSIDE each
                # replicate (correlation from the bootstrap sample + fresh MC draw), so the ratio
                # CI carries null-estimation uncertainty too (was: null held fixed -> CIs too narrow).
                Rho_b = normal_scores_corr(Rb)
                Zb = brng.standard_normal((N_MC_BOOT, N)) @ np.linalg.cholesky(
                    Rho_b + 1e-10 * np.eye(N)).T
                cop_b = copula_joint(Zb, mass, k, N)
                bemp.append(eb); brat.append(eb / cop_b if cop_b > 0 else np.nan)
            elo, ehi = np.nanpercentile(bemp, [2.5, 97.5])
            rlo, rhi = np.nanpercentile(brat, [2.5, 97.5])
            verdict = ("CLUSTERS MORE" if rlo > 1.0 else
                       "clusters less" if rhi < 1.0 else "no significant difference")
            print(f"{q:>6.3f} {k:>2} | {mass.mean():>8.4f} | {emp:>10.6f} [{elo:>8.6f},{ehi:>8.6f}] | "
                  f"{cop:>10.6f} {gld:>9.6f} {ind:>9.6f} {mps_p:>9.6f} | "
                  f"{emp/cop:>8.2f} [{rlo:>5.2f},{rhi:>5.2f}] | {verdict}", flush=True)
            results.append(dict(book=name, q=q, k=k, set_mass=mass.mean(), emp=emp,
                                emp_lo=elo, emp_hi=ehi, copula=cop, old_gauss=gld,
                                indep=ind, mps=mps_p, mps_calib_relerr=calib,
                                ratio=emp / cop, ratio_lo=rlo, ratio_hi=rhi))
    if mps_fit is not None:
        cal = [r["mps_calib_relerr"] for r in results if np.isfinite(r["mps_calib_relerr"])]
        print(f"MPS tail-set marginal calibration: mean |model-emp|/emp = "
              f"{np.mean(cal):.1%} (over bin-grid q rows)")
    return results


def kway_mps(fit, sets, k, N):
    cores = fit._cores3d(); Lenv, Renv, Z = fit._environments(cores); vals = []
    for S in itertools.combinations(range(N), k):
        S = sorted(S); i0 = S[0]; M = cores[i0][:, sets[i0], :]
        La = np.einsum("ac,asb,csd->bd", Lenv[i0], M, M); last = i0
        for m_ in S[1:]:
            for f in range(last + 1, m_):
                La = np.einsum("ac,asb,csd->bd", La, cores[f], cores[f])
            Mm = cores[m_][:, sets[m_], :]
            La = np.einsum("ac,asb,csd->bd", La, Mm, Mm); last = m_
        vals.append(float(np.einsum("ac,ac->", La, Renv[last + 1]) / Z))
    return float(np.mean(vals))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--synthetic":
        # t5 marginals + EXACT Gaussian copula: the old null must scream "clusters more"
        # in the deep tail; the copula null must say "no significant difference" (ratio CI ∋ 1).
        rng = np.random.default_rng(42)
        N, T = 12, 5000
        A = rng.standard_normal((N, N)) * 0.3 + np.eye(N)
        Rho = A @ A.T; d = np.sqrt(np.diag(Rho)); Rho = Rho / np.outer(d, d)
        Z = rng.standard_normal((T, N)) @ np.linalg.cholesky(Rho).T
        from scipy.stats import t as tdist
        X = tdist.ppf(norm.cdf(Z), df=5) * 0.01
        analyze(X, "SYNTHETIC t5-marginals + Gaussian copula (truth: NO excess clustering)")
        print("\nDONE"); sys.exit(0)

    from mps_regime import data as Dd, mps, binning, dmrg
    PARQUET = sys.argv[1]
    n_assets = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    chi = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    rets_full = Dd.to_returns(pd.read_parquet(PARQUET).astype(float), kind="log")
    _b, _, order = binning.discretize(rets_full, n_bins=6)
    rets = rets_full[order]
    if n_assets:
        rets = rets.iloc[:, :n_assets]
    print(f"BOOK={PARQUET} | N={rets.shape[1]} | Fiedler order: {list(rets.columns)}", flush=True)
    print(f"fitting DMRG d={D_BINS} chi={chi} ...", flush=True)
    fit = dmrg.fit_dmrg(rets, d=D_BINS, chi=chi, sweeps=16)
    p1 = fit.one_site_marginals()
    R = rets.to_numpy(); iu = np.triu_indices(rets.shape[1], 1)
    print(f"fit done | corr-capture {fit.mps_correlation()[iu].mean()/np.corrcoef(R.T)[iu].mean()*100:.0f}%",
          flush=True)
    res = analyze(R, PARQUET, mps_fit=fit, edges=fit.bin_edges, p1=p1)
    out = PARQUET.replace("data/", "").replace(".parquet", "")
    pd.DataFrame(res).to_csv(f"deep_tail_copula_{out}.csv", index=False)
    print(f"\nSAVED deep_tail_copula_{out}.csv\nDONE", flush=True)
