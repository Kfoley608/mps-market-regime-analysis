"""MPS Born machine for multi-asset return distributions.

Reproduces the core of the sponsor's "Multi-Asset Correlation Analysis": train an MPS on discretized
daily returns, then read off MPS-implied per-asset volatility, the correlation matrix, and the
tail-dependence matrix by generating synthetic scenarios from the fitted model.

Model (open-boundary, real tensors): P(v) = psi(v)^2 / Z, with
    psi(v) = A0[v0] . A1[:,v1,:] . ... . A_{N-1}[:,v_{N-1}]
Training minimizes the average negative log-likelihood by autodiff (Adam) — equivalent objective
to the platform's ALS sweeps. Z is contracted exactly via transfer matrices. Sampling is exact
(site-by-site "perfect sampling"), not MCMC.

All contractions accumulate a running log-scale so the N~20-site products stay numerically stable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


# --------------------------------------------------------------------------- #
# Core tensor contractions (torch, differentiable)                             #
# --------------------------------------------------------------------------- #
def _init_cores(n_sites, d, chi, seed=0):
    g = torch.Generator().manual_seed(seed)
    cores = [0.1 * torch.randn(d, chi, generator=g)]
    for _ in range(n_sites - 2):
        # near-identity on the bond + small noise = stable starting point
        mid = torch.eye(chi).unsqueeze(1).repeat(1, d, 1) / np.sqrt(d)
        mid = mid + 0.05 * torch.randn(chi, d, chi, generator=g)
        cores.append(mid)
    cores.append(0.1 * torch.randn(chi, d, generator=g))
    return [c.clone().detach().requires_grad_(True) for c in cores]


def _log_abs_psi(cores, X):
    """log|psi(v)| for a batch X (n, N) of integer bin indices, with log-scale accumulation."""
    n_sites = len(cores)
    vec = cores[0][X[:, 0]]                                   # (n, chi)
    nrm = vec.norm(dim=1, keepdim=True).clamp_min(1e-30)
    logs = torch.log(nrm[:, 0])
    vec = vec / nrm
    for k in range(1, n_sites - 1):
        Ck = cores[k].permute(1, 0, 2)[X[:, k]]              # (n, chi, chi)
        vec = torch.einsum("nl,nlr->nr", vec, Ck)
        nrm = vec.norm(dim=1, keepdim=True).clamp_min(1e-30)
        logs = logs + torch.log(nrm[:, 0])
        vec = vec / nrm
    last = cores[-1].t()[X[:, -1]]                            # (n, chi)
    psi = (vec * last).sum(1)                                 # (n,)
    return torch.log(psi.abs().clamp_min(1e-30)) + logs


def _log_Z(cores):
    """log of Z = <psi|psi>, via transfer matrices with log-scale accumulation."""
    n_sites = len(cores)
    E = cores[0].t() @ cores[0]                               # (chi, chi)
    nrm = E.abs().max().clamp_min(1e-30)
    logz = torch.log(nrm)
    E = E / nrm
    for k in range(1, n_sites - 1):
        E = torch.einsum("ac,asb,csd->bd", E, cores[k], cores[k])
        nrm = E.abs().max().clamp_min(1e-30)
        logz = logz + torch.log(nrm)
        E = E / nrm
    z = torch.einsum("ac,as,cs->", E, cores[-1], cores[-1])
    return torch.log(z.clamp_min(1e-30)) + logz


def _nll(cores, X):
    return _log_Z(cores) - 2.0 * _log_abs_psi(cores, X).mean()


# --------------------------------------------------------------------------- #
# Fitted-model container                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class MPSFit:
    cores: list                  # list of numpy arrays (trained tensors)
    n_sites: int
    d: int
    chi: int
    bin_values: np.ndarray       # (N, d) representative (mean) return per (asset, bin)
    bin_std: np.ndarray          # (N, d) empirical within-bin std (for variance-matched within-bin noise)
    bin_edges: list              # per-asset interior quantile edges
    columns: list                # asset names, in chain order
    nll_history: list

    def _torch_cores(self):
        return [torch.tensor(c, dtype=torch.float64) for c in self.cores]

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        cores = self._torch_cores()
        Xt = torch.tensor(X, dtype=torch.long)
        with torch.no_grad():
            lp = 2.0 * _log_abs_psi(cores, Xt) - _log_Z(cores)
        return lp.numpy()

    def sample_bins(self, n: int, seed: int = 0) -> np.ndarray:
        """Exact site-by-site sampling of n bin-configurations (n, N)."""
        rng = np.random.default_rng(seed)
        cores = [c.astype(np.float64) for c in self.cores]
        N, chi = self.n_sites, self.chi

        # right double-layer environments R[k] (chi x chi) for k = 1 .. N-1
        R = [None] * (N + 1)
        last = cores[-1]                                     # (chi, d)
        R[N - 1] = last @ last.T                             # (chi, chi)
        for k in range(N - 2, 0, -1):
            R[k] = np.einsum("asb,bd,csd->ac", cores[k], R[k + 1], cores[k])

        out = np.zeros((n, N), dtype=np.int64)
        A0 = cores[0]                                        # (d, chi)
        w = np.einsum("sb,bd,sd->s", A0, R[1], A0)           # (d,)
        p = np.clip(w, 0, None); p = p / p.sum()
        out[:, 0] = rng.choice(self.d, size=n, p=p)
        lvec = A0[out[:, 0]]                                 # (n, chi)
        for k in range(1, N - 1):
            Ck = cores[k]                                    # (chi, d, chi)
            cand = np.einsum("nl,lsr->nsr", lvec, Ck)        # (n, d, chi)
            w = np.einsum("nsr,rd,nsd->ns", cand, R[k + 1], cand)   # (n, d)
            w = np.clip(w, 0, None)
            w = w / w.sum(axis=1, keepdims=True)
            u = rng.random(n)
            choice = (np.cumsum(w, axis=1) > u[:, None]).argmax(axis=1)
            out[:, k] = choice
            lvec = cand[np.arange(n), choice]               # (n, chi)
        AL = cores[-1]                                       # (chi, d)
        wlast = (lvec @ AL) ** 2                             # (n, d)
        wlast = wlast / wlast.sum(axis=1, keepdims=True)
        u = rng.random(n)
        out[:, N - 1] = (np.cumsum(wlast, axis=1) > u[:, None]).argmax(axis=1)
        return out

    def sample_returns(self, n: int, seed: int = 0, within_bin_noise: bool = True) -> np.ndarray:
        """Generate n synthetic return vectors (n, N) from sampled bins.

        within_bin_noise=True (default, matches FabriQ's "uniform noise within each bin"): add uniform
        noise to each bin mean, scaled to that bin's EMPIRICAL within-bin std so the restored spread is
        variance-correct (Uniform[-sqrt(3)·s, sqrt(3)·s] has variance s^2). Using the empirical std
        rather than the raw bin width avoids hugely overstating the open fat-tail bins.
        False: map to the bin mean (the older, slightly vol-biased mapping).
        """
        bins = self.sample_bins(n, seed=seed)
        cols = np.arange(self.n_sites)
        if not within_bin_noise:
            return self.bin_values[cols, bins]              # (n, N) bin means
        rng = np.random.default_rng(seed + 12345)
        mu = self.bin_values[cols, bins]
        sd = self.bin_std[cols, bins]
        return mu + (rng.random(bins.shape) * 2 - 1) * np.sqrt(3.0) * sd   # variance-matched uniform

    # ---- exact marginals ("order-2" read-outs, no sampling noise) ----
    def _cores3d(self):
        cs = [self.cores[0].reshape(1, self.d, self.chi).astype(np.float64)]
        for k in range(1, self.n_sites - 1):
            cs.append(self.cores[k].astype(np.float64))
        cs.append(self.cores[-1].reshape(self.chi, self.d, 1).astype(np.float64))
        return cs

    def _environments(self, cores):
        N = self.n_sites
        Lenv = [None] * (N + 1); Lenv[0] = np.ones((1, 1))
        for k in range(N):
            Lenv[k + 1] = np.einsum("ac,asb,csd->bd", Lenv[k], cores[k], cores[k])
        Renv = [None] * (N + 1); Renv[N] = np.ones((1, 1))
        for k in range(N - 1, -1, -1):
            Renv[k] = np.einsum("asb,bd,csd->ac", cores[k], Renv[k + 1], cores[k])
        return Lenv, Renv, float(Lenv[N][0, 0])

    def one_site_marginals(self) -> np.ndarray:
        """Exact P(v_j) for every asset -> (N, d) array (rows sum to 1)."""
        cores = self._cores3d()
        Lenv, Renv, Z = self._environments(cores)
        out = np.zeros((self.n_sites, self.d))
        for j in range(self.n_sites):
            for s in range(self.d):
                M = cores[j][:, s, :]
                out[j, s] = np.einsum("ac,ab,bd,cd->", Lenv[j], M, Renv[j + 1], M)
        return out / Z

    def two_site_marginal(self, i: int, j: int, cores=None, env=None) -> np.ndarray:
        """Exact P(v_i, v_j) -> (d, d). i < j."""
        if cores is None:
            cores = self._cores3d()
        if env is None:
            env = self._environments(cores)
        Lenv, Renv, Z = env
        p = np.zeros((self.d, self.d))
        for si in range(self.d):
            La = np.einsum("ac,ab,cd->bd", Lenv[i], cores[i][:, si, :], cores[i][:, si, :])
            for k in range(i + 1, j):
                La = np.einsum("ac,asb,csd->bd", La, cores[k], cores[k])
            for sj in range(self.d):
                M = cores[j][:, sj, :]
                p[si, sj] = np.einsum("ac,ab,bd,cd->", La, M, Renv[j + 1], M)
        return p / Z

    def mps_volatility(self, annualize=True, within_bin_var=False) -> np.ndarray:
        """MPS-implied per-asset volatility from the exact 1-site marginals.

        with within_bin_var=True, adds the within-bin variance that quantile-binning discards (law of
        total variance) — the exact, noise-free equivalent of FabriQ's "uniform noise within each bin"
        scenario step. Uses each bin's EMPIRICAL within-bin variance, which restores the lost spread
        without overstating the fat-tail bins. Removes the small downward vol bias from bin means alone.
        """
        p1 = self.one_site_marginals()
        bv = self.bin_values
        mu = (p1 * bv).sum(1)
        var = (p1 * bv ** 2).sum(1) - mu ** 2
        if within_bin_var:
            var = var + (p1 * self.bin_std ** 2).sum(1)       # E[within-bin variance]
        vol = np.sqrt(np.clip(var, 0, None))
        return vol * np.sqrt(252) if annualize else vol

    def mps_correlation(self) -> np.ndarray:
        """MPS-implied (order-2) correlation matrix from exact 2-site marginals."""
        cores = self._cores3d()
        env = self._environments(cores)
        p1 = self.one_site_marginals()
        bv = self.bin_values
        mu = (p1 * bv).sum(1)
        var = (p1 * bv ** 2).sum(1) - mu ** 2
        N = self.n_sites
        C = np.eye(N)
        for i in range(N):
            for j in range(i + 1, N):
                p2 = self.two_site_marginal(i, j, cores, env)
                exy = (p2 * np.outer(bv[i], bv[j])).sum()
                cov = exy - mu[i] * mu[j]
                C[i, j] = C[j, i] = cov / np.sqrt(var[i] * var[j] + 1e-30)
        return C

    def mps_tail_dependence(self, q: float = 0.10) -> np.ndarray:
        """MPS-implied lower-tail dependence matrix: P(both in lower-q bins)/q, from 2-site
        marginals. Diagonal set to 1."""
        cores = self._cores3d()
        env = self._environments(cores)
        p1 = self.one_site_marginals()
        # lower-tail bin set per asset: smallest bins whose cumulative mass first reaches q
        lower = []
        for j in range(self.n_sites):
            cum = np.cumsum(p1[j])
            k = int(np.searchsorted(cum, q) + 1)
            lower.append(set(range(min(k, self.d))))
        N = self.n_sites
        T = np.eye(N)
        for i in range(N):
            for j in range(i + 1, N):
                p2 = self.two_site_marginal(i, j, cores, env)
                joint = sum(p2[a, b] for a in lower[i] for b in lower[j])
                T[i, j] = T[j, i] = joint / q
        return T

    # ---- regime diagnostics (the rolling-layer features) ----
    def entanglement_entropy(self, cut: int | None = None) -> float:
        """Von Neumann entanglement entropy (nats) across a chain cut, from the bond spectrum.

        The squared Schmidt values at the bond between site cut-1 and cut are the eigenvalues of
        Lenv[cut] @ Renv[cut] (product of the left/right Gram environments); S = -sum p ln p.
        (S <= ln chi).

        WARNING - NOT PROBABILITY-IDENTIFIED (2026-07 review, accepted): this is the entropy of
        the SIGNED fitted state. The training objective observes |psi| only, so two fits with
        identical P = psi^2/Z can report different S (e.g. uniform P: product state S=0 vs
        Hadamard S=ln2). Use as a fit diagnostic, not a market feature; the identified
        cut-dependence gauge is features.block_mutual_information. What P does identify is the
        floor S >= I_P(L;R)/2. See docs/Response-to-Review.md."""
        cores = self._cores3d()
        Lenv, Renv, _ = self._environments(cores)
        if cut is None:
            cut = self.n_sites // 2
        ev = np.linalg.eigvals(Lenv[cut] @ Renv[cut]).real
        ev = np.clip(ev, 0.0, None)
        p = ev / ev.sum()
        p = p[p > 1e-12]
        return float(-(p * np.log(p)).sum())

    def entanglement_profile(self) -> np.ndarray:
        """Entanglement entropy at every cut (1..N-1) — the localization signal."""
        return np.array([self.entanglement_entropy(k) for k in range(1, self.n_sites)])

    def tail_state_probability(self) -> float:
        """Model probability of the joint-extreme corner (every asset in its lowest bin)."""
        cfg = np.zeros((1, self.n_sites), dtype=np.int64)
        return float(np.exp(self.log_prob(cfg))[0])

    def encode(self, returns_block) -> np.ndarray:
        """Bin a returns block (DataFrame, columns in chain order) using THIS fit's bin edges."""
        arr = returns_block[self.columns].to_numpy()
        out = np.zeros(arr.shape, dtype=np.int64)
        for j in range(self.n_sites):
            out[:, j] = np.digitize(arr[:, j], self.bin_edges[j])
        return out

    def save(self, path) -> None:
        """Serialize this trained fit (pickle) so results are reproducible without retraining.

        Pure persistence — writes the already-trained tensors + metadata, never re-touches training.
        Note: pickle is tied to this class definition (fine for in-project reuse; for long-term archives
        prefer re-deriving from the dated data snapshot).
        """
        import pathlib
        import pickle
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)


# --------------------------------------------------------------------------- #
# Discretization + training entry point                                        #
# --------------------------------------------------------------------------- #
def _discretize(returns, d):
    """Quantile-bin each asset into d bins; return integer codes, edges, bin-mean returns, and per-bin
    [lo, hi] ranges. Interior bin bounds are the quantile edges; the open tail bins are capped by the
    empirical min/max — these ranges drive the within-bin uniform sampling that matches FabriQ."""
    arr = returns.to_numpy()
    n, N = arr.shape
    codes = np.zeros((n, N), dtype=np.int64)
    edges, bin_values = [], np.zeros((N, d))
    bin_std = np.zeros((N, d))
    qs = np.linspace(0, 1, d + 1)[1:-1]
    for j in range(N):
        e = np.quantile(arr[:, j], qs)
        c = np.digitize(arr[:, j], e)
        codes[:, j] = c
        edges.append(e)
        for b in range(d):
            vals = arr[c == b, j]
            bin_values[j, b] = vals.mean() if len(vals) else 0.0
            bin_std[j, b] = vals.std() if len(vals) > 1 else 0.0   # empirical within-bin spread
    return codes, edges, bin_values, bin_std


def fit_born_machine(returns, d=16, chi=8, sweeps=400, lr=0.05, seed=0, verbose=False,
                     init_cores=None, l2=0.0) -> MPSFit:
    """Train an MPS Born machine on a daily-returns DataFrame (columns = assets, in chain order).

    `d` = physical dimension (bins/asset), `chi` = bond dimension — defaults match the sponsor (d=16, chi=8).
    `sweeps` = Adam optimization steps over the full sample. Pass `init_cores` (a previous fit's
    `.cores`) to WARM-START — much faster for rolling re-fits and a strong prior under data scarcity.
    `l2` = optional L2 regularization (Adam weight decay) on the tensor entries — a dormant hook
    (default 0.0 = no effect). Turn it on only if a window overfits (train NLL << held-out NLL),
    which is most likely on small/sparse windows at high d; analogous to FabriQ's regularized lstsq.
    """
    codes, edges, bin_values, bin_std = _discretize(returns, d)
    N = returns.shape[1]
    if init_cores is None:
        cores = _init_cores(N, d, chi, seed=seed)
    else:
        cores = [torch.tensor(c, dtype=torch.float32).clone().requires_grad_(True) for c in init_cores]
    Xt = torch.tensor(codes, dtype=torch.long)
    opt = torch.optim.Adam(cores, lr=lr, weight_decay=l2)   # weight_decay=0 -> no regularization (default)
    history = []
    for step in range(sweeps):
        opt.zero_grad()
        loss = _nll(cores, Xt)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        if verbose and step % 50 == 0:
            print(f"step {step:4d}  NLL {history[-1]:.4f}")
    return MPSFit(
        cores=[c.detach().numpy() for c in cores],
        n_sites=N, d=d, chi=chi,
        bin_values=bin_values, bin_std=bin_std, bin_edges=edges,
        columns=list(returns.columns), nll_history=history,
    )


def rolling_mps_features(returns, window=250, step=10, d=3, chi=8, sweeps=200, recent=10,
                         warm_start=True, verbose=False, l2=0.0):
    """Roll the MPS over the return panel and extract regime diagnostics per window.

    This is the core of the regime layer: refit the MPS on each trailing window (warm-started from
    the previous one) and read off the time-varying complexity diagnostics. Returns a DataFrame
    indexed by window-END date with columns (the sponsor's candidate-feature set):
        window_nll  : average NLL of the window (overall fit difficulty)
        recent_nll  : NLL of the most-recent `recent` days under this fit (surprise / change-point) [F5]
        ent_mid     : entanglement entropy at the central cut (total cross-block dependence)        [F2]
        ent_mean    : mean entanglement entropy over all cuts                                        [F3]
        tail_prob   : joint-extreme probability (all assets in their lowest bin)                     [F6]
        eff_chi     : effective bond dimension = exp(ent_mid) — 'chi required' complexity dial       [F1]
        recon_err   : MPS-vs-empirical correlation error (fit quality / structure capture)           [F9]
        mi_adj      : mean adjacent-pair mutual information (nonlinear dependence)                    [F4]
        drift       : relative L2 change in tensors vs the previous window (regime-break signal)     [F7]

    (Ordering sensitivity [F8] and the exact required-chi [F1] refit per ordering/chi and are run
    on a coarse cadence via features.py, not here. compression_ratio [F10] is a structural constant.)

    Defaults are the data-starved-window profile (coarse d=3) — appropriate for ~250-day refits.
    """
    from . import features as F
    idx = returns.index
    rows, dates = [], []
    prev_fit = None
    for end in range(window, len(idx) + 1, step):
        block = returns.iloc[end - window:end]
        fit = fit_born_machine(block, d=d, chi=chi, sweeps=sweeps, lr=0.03, seed=0,
                               init_cores=prev_fit.cores if (warm_start and prev_fit) else None, l2=l2)
        recent_codes = fit.encode(block.iloc[-recent:])
        ent_mid = fit.entanglement_entropy()
        rows.append({
            "window_nll": fit.nll_history[-1],
            "recent_nll": float(-fit.log_prob(recent_codes).mean()),
            "ent_mid": ent_mid,
            "ent_mean": float(fit.entanglement_profile().mean()),
            "tail_prob": fit.tail_state_probability(),
            "eff_chi": float(np.exp(ent_mid)),
            "recon_err": F.reconstruction_error(fit, block),
            "mi_adj": F.mean_mutual_information(fit, adjacent=True),
            "drift": F.parameter_drift(fit, prev_fit) if prev_fit is not None else float("nan"),
        })
        dates.append(idx[end - 1])
        prev_fit = fit
        if verbose:
            print(f"{idx[end-1].date()}  nll {rows[-1]['window_nll']:.2f}  ent {rows[-1]['ent_mid']:.3f}")
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))


def load_fit(path) -> MPSFit:
    """Load a previously saved MPSFit (see MPSFit.save) — reproduces the trained model without retraining."""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
