"""Two-site DMRG / ALS trainer for the MPS Born machine (canonical-form fallback to Adam).

Why this file exists
--------------------
The global-autodiff (Adam) trainer in `mps.py` fits the per-asset marginals but badly
UNDER-fits cross-asset correlation, and it gets worse with N (it reaches only ~2-8% of the
achievable off-diagonal correlation at N>=12). That is the textbook failure mode of GLOBAL
gradient descent on an MPS: the loss landscape is ill-conditioned because the normalization Z
couples every tensor through a full transfer-matrix contraction.

The fix (Han, Wang, Fan, Wang, Zhang, PRX 2018, "Unsupervised Generative Modeling Using MPS")
is two-site DMRG with LOCAL sweeps in MIXED CANONICAL FORM:

  - Keep tensors left of the orthogonality center left-canonical and those to the right
    right-canonical. Then Z = <psi|psi> = ||center||^2 = ||T||^2 (Frobenius), with NO global
    contraction. The local update is therefore perfectly conditioned.
  - At each bond (k,k+1) merge the two cores into a two-site tensor T, take a gradient-ascent
    step on the per-sample log-likelihood (Z = ||T||^2 locally), then SVD T, truncate to chi,
    and split: the U side becomes the new (left- or right-) canonical core and S*Vh (or U*S)
    becomes the new center at the next bond. Per-sample LEFT/RIGHT environment vectors are
    cached and updated incrementally as the center moves.

This module does NOT touch mps.py. It reuses `_discretize` (binning) and returns an `MPSFit`
so every read-off (mps_correlation, mps_volatility, one_site_marginals, log_prob, ...) works
unchanged. The local gradient is taken with PyTorch autodiff on T — that part is not the hard
part; the hard part (which the earlier "block Adam + SVD" hack got wrong) is the canonical form
and the environment caching, both of which are done explicitly here.

Convention (matches mps.py):
    cores[0]      : (d, chi)
    cores[k]      : (chi, d, chi)   for 0 < k < N-1
    cores[-1]     : (chi, d)
    psi(v) = A0[v0] @ A1[:,v1,:] @ ... @ A_{N-1}[:,v_{N-1}];   P = psi^2 / Z.

Internally we work with a uniform list of rank-3 cores of shape (Dl, d, Dr) (with Dl=1 on the
first and Dr=1 on the last) so the canonical bookkeeping is one code path; we squeeze back to the
mps.py convention only when packaging the final MPSFit.
"""

from __future__ import annotations

import numpy as np
import torch

from .mps import MPSFit, _discretize


# --------------------------------------------------------------------------- #
# Canonical-form helpers (all on uniform rank-3 cores of shape (Dl, d, Dr))    #
# --------------------------------------------------------------------------- #
def _right_canonicalize(A: np.ndarray):
    """LQ a core (Dl, d, Dr) into a RIGHT-canonical core Q (sum_s Q_s Q_s^T = I) and the carry
    matrix L absorbed into the core to its left.  Returns (L, Q)."""
    Dl, d, Dr = A.shape
    M = A.reshape(Dl, d * Dr)
    # LQ via QR of the transpose:  M^T = Qb Rb  ->  M = Rb^T Qb^T,  L = Rb^T, Q = Qb^T
    Qb, Rb = np.linalg.qr(M.T)
    k = Qb.shape[1]
    Lmat = Rb.T                                # (Dl, k)
    Q = Qb.T.reshape(k, d, Dr)                 # right-canonical
    return Lmat, Q


def _make_right_canonical_from_center0(cores):
    """Bring the chain into mixed-canonical form with the orthogonality center at site 0:
    every core 1..N-1 is RIGHT-canonical, core 0 holds the norm.  Returns new list."""
    out = [c.copy() for c in cores]
    carry = None
    for k in range(len(out) - 1, 0, -1):
        if carry is not None:
            out[k] = np.einsum("asb,bc->asc", out[k], carry)
        Lmat, Q = _right_canonicalize(out[k])
        out[k] = Q
        carry = Lmat
    out[0] = np.einsum("asb,bc->asc", out[0], carry)
    return out


# --------------------------------------------------------------------------- #
# Init                                                                          #
# --------------------------------------------------------------------------- #
def _init_uniform_cores(N, d, chi, seed=0):
    """Random rank-3 cores (Dl, d, Dr) with growing/shrinking bond at the chain ends so we never
    carry a bond larger than is representable (Dl, Dr capped by the cumulative physical dim)."""
    rng = np.random.default_rng(seed)
    # bond dimension at each cut, capped by chi and by d**(distance from the nearer end)
    bonds = [1]
    for k in range(1, N):
        left_cap = min(chi, d ** k)
        right_cap = min(chi, d ** (N - k))
        bonds.append(int(min(left_cap, right_cap, chi)))
    bonds.append(1)
    cores = []
    for k in range(N):
        Dl, Dr = bonds[k], bonds[k + 1]
        cores.append(0.1 * rng.standard_normal((Dl, d, Dr)))
    return cores


# --------------------------------------------------------------------------- #
# Two-site DMRG sweep                                                           #
# --------------------------------------------------------------------------- #
def _twosite_gradstep(T0, Lvecs, Rvecs, bk, bk1, n_inner, lr, l2, solver="adam",
                      weights=None):
    """One (or a few) gradient-ASCENT steps on the local log-likelihood for the merged two-site
    tensor T (shape Dl, d, d, Dr), in canonical form so Z = ||T||^2.

        log L = (2/Ns) * sum_samples w_s log|psi_s| / mean(w)  -  log Z,
        psi_s = Lvec_s @ T[:, bk_s, bk1_s, :] @ Rvec_s,   Z = ||T||_F^2.

    `weights` (optional, Ns) turn this into a weighted-MLE local step — the M-step of an EM over
    mixture/Markov-switching Born machines (posterior responsibilities as weights). None = uniform.

    Implemented with torch autodiff (correct + less bug-prone); canonical form makes it well
    conditioned.  l2 adds a tiny ridge on T (Frobenius) to stabilize near-degenerate bonds."""
    Dl, d, _, Dr = T0.shape
    T = torch.tensor(T0, dtype=torch.float64, requires_grad=True)
    L = torch.tensor(Lvecs, dtype=torch.float64)          # (Ns, Dl)
    R = torch.tensor(Rvecs, dtype=torch.float64)          # (Ns, Dr)
    bk = torch.tensor(bk, dtype=torch.long)
    bk1 = torch.tensor(bk1, dtype=torch.long)
    w = None
    if weights is not None:
        w = torch.tensor(np.asarray(weights, dtype=np.float64), dtype=torch.float64)
        w = w / w.mean().clamp_min(1e-300)                # normalized so the scale matches unweighted

    def closure_loss():
        slc = T[:, bk, bk1, :]                            # (Dl, Ns, Dr)
        psi = torch.einsum("sa,asb,sb->s", L, slc, R)     # (Ns,)
        logZ = torch.log((T * T).sum().clamp_min(1e-300))
        lp = torch.log(psi.abs().clamp_min(1e-300))
        ll = 2.0 * (lp.mean() if w is None else (w * lp).mean()) - logZ
        return -ll + l2 * (T * T).sum()

    last = None
    if solver == "lbfgs":
        # LBFGS converges the (well-conditioned, canonical) local subproblem in a handful of
        # iterations -> each two-site update reaches its conditional optimum, so correlation
        # builds up per-sweep instead of crawling.
        opt = torch.optim.LBFGS([T], lr=lr, max_iter=n_inner, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = closure_loss()
            loss.backward()
            return loss

        loss = opt.step(closure)
        last = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
    else:
        opt = torch.optim.Adam([T], lr=lr)
        for _ in range(n_inner):
            opt.zero_grad()
            loss = closure_loss()
            loss.backward()
            opt.step()
            last = float(loss.detach())
    T_out = T.detach().numpy()
    if not np.isfinite(T_out).all():   # degenerate local step (e.g. chi=1 corner cases):
        return T0, last               # skip this update rather than poison the SVD
    return T_out, last


def _split_two_site(T, chi, direction, eps=1e-12):
    """SVD-split a two-site tensor T (Dl, d, d, Dr) into two rank-3 cores, truncating to <=chi.

    direction='right': left core becomes LEFT-canonical (U), center = S*Vh -> goes to the right.
    direction='left' : right core becomes RIGHT-canonical (Vh), center = U*S -> goes to the left.
    Returns (Acore, Bcore, kept_singular_values)."""
    Dl, d, _, Dr = T.shape
    M = T.reshape(Dl * d, d * Dr)
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    # truncate
    keep = S > eps * (S[0] if S.size else 1.0)
    k = int(min(chi, max(1, keep.sum())))
    U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
    if direction == "right":
        A = U.reshape(Dl, d, k)                            # left-canonical
        B = (np.diag(S) @ Vh).reshape(k, d, Dr)           # center
    else:
        A = (U @ np.diag(S)).reshape(Dl, d, k)            # center
        B = Vh.reshape(k, d, Dr)                           # right-canonical
    return A, B, S


def fit_dmrg(returns, d=8, chi=8, sweeps=16, n_inner=20, lr=1.0, l2=1e-8, seed=0,
             verbose=False, init_cores=None, solver="lbfgs", adam_warmup=0,
             weights=None, discretization=None) -> MPSFit:
    """Train an MPS Born machine on a daily-returns DataFrame via two-site DMRG in canonical form.

    Parameters
    ----------
    returns : DataFrame (columns = assets in chain order)
    d, chi  : physical dim (bins/asset) and bond dimension.
    sweeps  : number of full left->right->left DMRG sweeps over all bonds.
    n_inner : inner optimizer iterations per two-site update.
    lr      : inner optimizer learning rate (LBFGS step; line search handles sizing, 1.0 is fine).
    l2      : tiny Frobenius ridge on the local two-site tensor (stability).
    solver  : "lbfgs" (default — converges each local subproblem, so correlation builds up per
        sweep) or "adam" (slower, simpler).
    adam_warmup : if >0 and no init_cores given, first fit `adam_warmup` Adam steps and start DMRG
        from them. DEFAULT 0 (pure cold DMRG) — empirically a cold random start finds MORE
        correlation than warm-starting from Adam, because the near-uncorrelated Adam optimum is a
        poor DMRG basin (at N=24, cold reaches 0.18 off-diag corr vs 0.01 when warm-started from
        Adam). Kept as a hook only.
    init_cores : optional warm-start cores in the mps.py convention (overrides adam_warmup).

    weights : optional (Ns,) nonnegative per-sample weights (EM posterior responsibilities).
        The fit becomes weighted MLE; None = uniform.
    discretization : optional (codes, edges, bin_values, bin_std) tuple to REUSE a shared
        binning (e.g. all regime emissions of a Markov-switching model must share one grid so
        their likelihoods are comparable). None = discretize `returns` here.

    Returns an `MPSFit` (same class as mps.py) so all read-offs work unchanged.
    """
    if discretization is not None:
        codes, edges, bin_values, bin_std = discretization
    else:
        codes, edges, bin_values, bin_std = _discretize(returns, d)
    N = returns.shape[1]
    X = codes.astype(np.int64)                            # (Ns, N) data bins
    Ns = X.shape[0]

    # ---- optional Adam warm-up (locks in marginals; DMRG then adds the correlation) ----
    if init_cores is None and adam_warmup and adam_warmup > 0:
        from .mps import fit_born_machine
        init_cores = fit_born_machine(returns, d=d, chi=chi, sweeps=adam_warmup,
                                      lr=0.05, seed=seed).cores

    # ---- init + bring into mixed-canonical form, orthogonality center at site 0 ----
    if init_cores is None:
        cores = _init_uniform_cores(N, d, chi, seed=seed)
    else:
        # lift mps.py-convention cores (d,chi)/(chi,d,chi)/(chi,d) to uniform (Dl,d,Dr)
        cores = [np.asarray(init_cores[0]).reshape(1, d, -1).astype(np.float64)]
        for k in range(1, N - 1):
            cores.append(np.asarray(init_cores[k]).astype(np.float64))
        cores.append(np.asarray(init_cores[-1]).reshape(-1, d, 1).astype(np.float64))
    cores = _make_right_canonical_from_center0(cores)     # cores[1..] right-canonical, center @ 0

    # Per-sample environment vectors are products of ~N matrices; over a long chain (N>=~16) they
    # UNDERFLOW geometrically toward 0 (mean|R| ~ 1e-9 by site 3 at N=24), which zeros out psi_s and
    # kills the local gradient at the far end -> no correlation learned.  Fix: row-normalize each
    # env vector to unit L2 norm.  This is EXACT for the local update: psi_s = L_s.T_slice.R_s, and
    # the objective uses psi'_s/psi_s (scale-free) with Z=||T||^2 (canonical, env-independent), so a
    # per-sample rescale of L_s or R_s only adds a T-independent constant to the local log-likelihood
    # and leaves argmax_T unchanged.
    def _rownorm(V):
        return V / np.linalg.norm(V, axis=1, keepdims=True).clip(1e-300)

    # ---- per-sample RIGHT environments R[k] (Ns, D_k) for k=1..N (row-normalized) ----
    Rcache = [None] * (N + 1)
    Rcache[N] = np.ones((Ns, 1))
    for k in range(N - 1, 0, -1):
        sel = cores[k][:, X[:, k], :]                     # (Dl, Ns, Dr)
        Rcache[k] = _rownorm(np.einsum("asb,sb->sa", sel, Rcache[k + 1]))

    # left env at bond 0 is trivial (Dl=1)
    Lcache = [None] * N
    Lcache[0] = np.ones((Ns, 1))                          # L[k][s, a] : (Ns, D_k)


    history = []

    def sweep_lr():
        """Left -> right sweep over bonds 0..N-2.  Center starts at site 0."""
        for k in range(N - 1):
            T = np.einsum("asb,btc->astc", cores[k], cores[k + 1])   # (Dl,d,d,Dr)
            Lv, Rv = Lcache[k], Rcache[k + 2]
            T, loss = _twosite_gradstep(T, Lv, Rv, X[:, k], X[:, k + 1],
                                        n_inner, lr, l2, solver=solver, weights=weights)
            A, B, _ = _split_two_site(T, chi, direction="right")
            cores[k], cores[k + 1] = A, B
            history.append(loss)
            # move center to k+1: update LEFT env to include now-left-canonical core k
            selA = A[:, X[:, k], :]                                   # (Dl, Ns, Dk1)
            Lcache[k + 1] = _rownorm(np.einsum("sa,asb->sb", Lcache[k], selA))
        # after this sweep the center sits at site N-1

    def sweep_rl():
        """Right -> left sweep over bonds N-2..0.  Center starts at site N-1."""
        for k in range(N - 2, -1, -1):
            T = np.einsum("asb,btc->astc", cores[k], cores[k + 1])
            Lv, Rv = Lcache[k], Rcache[k + 2]
            T, loss = _twosite_gradstep(T, Lv, Rv, X[:, k], X[:, k + 1],
                                        n_inner, lr, l2, solver=solver, weights=weights)
            A, B, _ = _split_two_site(T, chi, direction="left")
            cores[k], cores[k + 1] = A, B
            history.append(loss)
            # move center to k: update RIGHT env to include now-right-canonical core k+1
            selB = B[:, X[:, k + 1], :]                               # (Dk1, Ns, Dr)
            Rcache[k + 1] = _rownorm(np.einsum("asb,sb->sa", selB, Rcache[k + 2]))

    def _true_nll():
        """Exact NLL of the current cores via mps.py's transfer-matrix Z (the honest objective).
        The per-sweep `history` value is the LOCAL canonical objective shifted by a per-sample
        constant from the env row-normalization, so it is not directly comparable across N — this
        recomputes the real number when asked."""
        from .mps import _log_abs_psi, _log_Z
        tc = ([torch.tensor(cores[0][0], dtype=torch.float64)]
              + [torch.tensor(cores[k], dtype=torch.float64) for k in range(1, N - 1)]
              + [torch.tensor(cores[-1][:, :, 0], dtype=torch.float64)])
        Xt = torch.tensor(X, dtype=torch.long)
        with torch.no_grad():
            return float(_log_Z(tc) - 2.0 * _log_abs_psi(tc, Xt).mean())

    for sw in range(sweeps):
        sweep_lr()
        sweep_rl()
        if verbose:
            print(f"sweep {sw:2d}  true-NLL {_true_nll():.4f}")

    # ---- package into mps.py convention: squeeze end bonds, normalize the center ----
    # The center currently sits at site 0 (after the last right->left sweep). Renormalize it so
    # Z = ||center||^2 = 1 (NLL invariant to single-core rescale; gives clean normalization).
    c0 = cores[0]
    nrm = np.sqrt((c0 * c0).sum())
    cores[0] = c0 / max(nrm, 1e-300)

    # MPSFit's read-offs (_cores3d) require UNIFORM bonds == chi: cores[0] (d,chi),
    # cores[k] (chi,d,chi), cores[-1] (chi,d).  DMRG keeps smaller bonds near the chain ends, so
    # we ZERO-PAD every bond up to chi.  Padding with zeros leaves the represented tensor (and
    # hence P) EXACTLY unchanged — the extra bond basis directions carry no weight.
    def _pad(A, Dl_t, Dr_t):
        Dl, d_, Dr = A.shape
        out = np.zeros((Dl_t, d_, Dr_t))
        out[:Dl, :, :Dr] = A
        return out

    cores = [_pad(c, chi, chi) for c in cores]
    # first core (chi, d, chi) -> keep only the live first row block then expose as (d, chi)
    packaged = []
    packaged.append(cores[0][0])                                             # (d, chi)
    for k in range(1, N - 1):
        packaged.append(cores[k])                                            # (chi, d, chi)
    packaged.append(cores[-1][:, :, 0])                                      # (chi, d)

    # final NLL under the exact (transfer-matrix) Z, for an apples-to-apples number vs Adam
    fit = MPSFit(
        cores=[c.astype(np.float64) for c in packaged],
        n_sites=N, d=d, chi=chi,
        bin_values=bin_values, bin_std=bin_std, bin_edges=edges,
        columns=list(returns.columns), nll_history=history,
    )
    final_nll = float(-fit.log_prob(X).mean())
    fit.nll_history = history + [final_nll]
    return fit
