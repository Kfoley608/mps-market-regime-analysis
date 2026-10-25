"""Production MPS model: overfitting-aware (chi chosen by held-out likelihood) + accurate (d=40 tail
resolution) + full feature set + higher-order battery vs empirical/Gaussian. Runs banks + semis.
Usage: python production_model.py"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, torch
from sklearn.metrics import mutual_info_score
from mps_regime import data as D, mps, binning, dmrg, features as F

BOOKS = {"banks": "data/banks_crsp_prices.parquet", "semis": "data/semis_crsp_prices.parquet"}
D_SELECT, SWEEPS_SEL = 12, 10     # light resolution to RANK chi by held-out dependence (fast)
D_FINAL, SWEEPS_FIN = 40, 12      # proper-resolution final production fit (accuracy)
CHI_GRID = [4, 8, 12, 16, 24]

def hom_samples(X, N, iu):
    Z = (X - X.mean(0)) / X.std(0)
    coskew = np.mean([(Z[:, i]**2 * Z[:, j]).mean() for i in range(N) for j in range(N) if i != j])
    cokurt = np.mean([(Z[:, i]**2 * Z[:, j]**2).mean() for i, j in zip(*iu)])
    def lam(sign):
        thr = np.quantile(sign*X, 0.90, axis=0); hi = sign*X > thr
        return np.mean([(hi[:, i] & hi[:, j]).mean()/0.10 for i, j in zip(*iu)])
    lamL, lamU = lam(-1), lam(+1)
    B = np.stack([np.digitize(X[:, k], np.quantile(X[:, k], np.linspace(0,1,9)[1:-1])) for k in range(N)], 1)
    mi = np.mean([mutual_info_score(B[:, i], B[:, j]) for i, j in zip(*iu)])
    return dict(lambda_L=lamL, tail_asym=lamL-lamU, co_skew=coskew, co_kurt=cokurt, mutual_info=mi)

def hom_mps(fit, N, iu):
    p1 = fit.one_site_marginals(); bv = fit.bin_values
    mu = (p1*bv).sum(1); sd = np.sqrt((p1*bv**2).sum(1)-mu**2)
    cs, ck = [], []
    for i, j in zip(*iu):
        p2 = fit.two_site_marginal(i, j); zi=(bv[i]-mu[i])/sd[i]; zj=(bv[j]-mu[j])/sd[j]
        ck.append(float((p2*np.outer(zi**2, zj**2)).sum()))
        cs.append(float((p2*np.outer(zi**2, zj)).sum())); cs.append(float((p2*np.outer(zi, zj**2)).sum()))
    return dict(lambda_L=float(fit.mps_tail_dependence(0.10)[iu].mean()),
                tail_asym=F.tail_dependence_asymmetry(fit),
                co_skew=float(np.mean(cs)), co_kurt=float(np.mean(ck)),
                mutual_info=float(np.mean([F.mutual_information(fit,i,j) for i,j in zip(*iu)])))

for name, pq in BOOKS.items():
    print(f"\n{'='*70}\nPRODUCTION MODEL: {name}\n{'='*70}", flush=True)
    rets_full = D.to_returns(pd.read_parquet(pq).astype(float), kind="log")
    T = len(rets_full)
    rng = np.random.default_rng(0); perm = rng.permutation(T); cut = int(0.7*T)
    tr, te = sorted(perm[:cut]), sorted(perm[cut:])
    # REVIEW FIX 2026-07-14 (BUG-LOG #15): Fiedler ordering AND bin edges from TRAIN ONLY; test rows
    # encoded with the train edges (same digitize convention as MPSFit.encode). Previously the codes
    # came from full-sample edges while the fit used train-only edges -> incompatible codes + leakage.
    _b, _, order = binning.discretize(rets_full.iloc[tr], n_bins=6); rets = rets_full[order]
    N = rets.shape[1]; iu = np.triu_indices(N, 1)
    X = rets.to_numpy()
    print(f"N={N} T={T} | Fiedler order (train-only)={list(order)}", flush=True)

    # ---- 1. chi selection by held-out likelihood (ANTI-OVERFITTING) ----
    codes_tr, edges_tr, _, _ = mps._discretize(rets.iloc[tr], D_SELECT)
    arr_te = rets.iloc[te].to_numpy()
    codes_te = np.column_stack([np.digitize(arr_te[:, k], edges_tr[k]) for k in range(N)])
    ind = 0.0
    for k in range(N):
        p = np.bincount(codes_tr[:,k], minlength=D_SELECT)+1.0; p/=p.sum(); ind += -np.log(p[codes_te[:,k]]).mean()
    print("chi-selection (held-out dependence captured; pick the MAX):", flush=True)
    best_chi, best_dep = CHI_GRID[0], -1e9
    for chi in CHI_GRID:
        f = dmrg.fit_dmrg(rets.iloc[tr], d=D_SELECT, chi=chi, sweeps=SWEEPS_SEL,
                          discretization=(codes_tr, edges_tr, None, None) if False else None)
        tc = [torch.tensor(c) for c in f.cores]
        nll_te = float(mps._nll(tc, torch.tensor(f.encode(rets.iloc[te]), dtype=torch.long)))
        nll_tr = float(mps._nll(tc, torch.tensor(f.encode(rets.iloc[tr]), dtype=torch.long)))
        dep = ind - nll_te
        star = " <-- best" if dep > best_dep else ""
        if dep > best_dep: best_dep, best_chi = dep, chi
        print(f"  chi={chi:2d}: train_nll={nll_tr:.3f} test_nll={nll_te:.3f} gap={nll_te-nll_tr:+.3f} heldout_dep={dep:+.3f}{star}", flush=True)
    print(f"  => SELECTED chi*={best_chi} (max held-out dependence; avoids overfitting)", flush=True)

    # ---- 2. final production fit at d=40, chi* on FULL data ----
    fit = dmrg.fit_dmrg(rets, d=D_FINAL, chi=best_chi, sweeps=SWEEPS_FIN)
    corr_cap = fit.mps_correlation()[iu].mean()/np.corrcoef(X.T)[iu].mean()*100
    print(f"\nFINAL MODEL: d={D_FINAL} chi*={best_chi} | corr-capture={corr_cap:.0f}%", flush=True)

    # ---- 3. candidate + new features ----
    print("FEATURES:", flush=True)
    print(f"  entanglement={F.entanglement_entropy(fit):.3f}  mean_MI={F.mean_mutual_information(fit,adjacent=False):.3f}"
          f"  tail_state_prob={F.tail_state_probability(fit):.4f}", flush=True)
    print(f"  total_correlation={F.total_correlation(fit):.3f}  tail_asymmetry={F.tail_dependence_asymmetry(fit):+.3f}"
          f"  corr_length={F.correlation_length(fit):.3f}", flush=True)
    print(f"  recon_error={F.reconstruction_error(fit,rets):.4f}  bond_dim@quality={F.bond_dimension_at_quality(fit):.2f}"
          f"  compression_ratio={F.compression_ratio(fit):.2f}", flush=True)

    # ---- 4. higher-order battery: empirical (truth) vs Gaussian vs MPS ----
    emp = hom_samples(X, N, iu)
    G = rng.multivariate_normal(X.mean(0), np.cov(X.T), size=200000); gau = hom_samples(G, N, iu)
    m = hom_mps(fit, N, iu)
    print("HIGHER-ORDER BATTERY (does MPS beat Gaussian at matching empirical truth?):", flush=True)
    print(f"  {'metric':12s} {'empirical':>10} {'Gaussian':>10} {'MPS':>10}  winner", flush=True)
    for k in ["lambda_L","tail_asym","co_skew","co_kurt","mutual_info"]:
        w = "MPS" if abs(m[k]-emp[k]) < abs(gau[k]-emp[k]) else "Gaussian"
        print(f"  {k:12s} {emp[k]:>10.4f} {gau[k]:>10.4f} {m[k]:>10.4f}  {w}", flush=True)
    print(f"### DONE {name} ###", flush=True)
print("\nALL DONE", flush=True)
