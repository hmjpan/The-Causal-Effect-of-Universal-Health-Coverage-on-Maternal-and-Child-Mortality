"""
Step 4: Core GSC algorithm — Bai (2009) Iterative Principal Components.
All paths via config.py. No hardcoded paths.
"""
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from config import PROC_DIR, YEAR_START, YEAR_END, COVARIATES, EXCLUDE_TREATED, EXCLUDE_NO_DATA


def estimate_ife(Y, X=None, r=1, max_iter=200, tol=1e-6, verbose=False):
    """
    Bai (2009) Interactive Fixed Effects via iterative principal components.
    Model: Y_it = X_it'*beta + lambda_i'*F_t + e_it

    Returns: beta (p,), F (T x r), Lambda (N x r), Y_hat (N x T)
    """
    N, T = Y.shape
    obs = ~np.isnan(Y)
    Y_fill = np.where(obs, Y, np.nanmean(Y))

    if X is None:
        has_X = False
        p = 0
        beta = None
    else:
        has_X = True
        p = X.shape[2]
        beta = np.zeros(p)

    # Init F from SVD
    Y_dm = Y_fill - np.nanmean(Y_fill)
    U, s_full, Vt = np.linalg.svd(Y_dm, full_matrices=False)
    r_eff = min(r, len(s_full))
    F = Vt[:r_eff].T.copy()
    Lambda = U[:, :r_eff] * s_full[:r_eff]
    # Normalize F'F/T = I
    sc = np.sqrt(np.diag(F.T @ F) / T)
    F = F / sc[np.newaxis, :]
    Lambda = Lambda * sc[np.newaxis, :]

    prev_ssr = np.inf

    for it in range(max_iter):
        # ---- Beta given F ----
        if has_X:
            R = Y_fill.copy()
            for _ in range(5):
                L_cur = R @ F / T
                fpart = L_cur @ F.T
                y_tgt = (R - fpart).ravel()
                X_stk = np.zeros((N * T, p))
                for k in range(p):
                    X_stk[:, k] = X[:, :, k].ravel()
                o_flat = obs.ravel()
                try:
                    beta = np.linalg.lstsq(X_stk[o_flat], y_tgt[o_flat], rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
                Xb = np.zeros((N, T))
                for k in range(p):
                    Xb += beta[k] * X[:, :, k]
                R = Y_fill - Xb
        else:
            R = Y_fill.copy()

        # ---- F from eigenvectors of R'R ----
        RR = R.T @ R / (N * T)
        try:
            w, v = eigh(RR)
            idx = np.argsort(w)[::-1][:r_eff]
            F_new = v[:, idx]
        except np.linalg.LinAlgError:
            F_new = F.copy()

        sc_f = np.sqrt(np.diag(F_new.T @ F_new) / T)
        sc_f[sc_f < 1e-10] = 1.0
        F_new = F_new / sc_f[np.newaxis, :]
        L_new = R @ F_new / T

        # ---- Convergence ----
        fpart_new = L_new @ F_new.T
        if has_X:
            Xb = np.zeros((N, T))
            for k in range(p):
                Xb += beta[k] * X[:, :, k]
            Yh = Xb + fpart_new
        else:
            Yh = fpart_new

        ssr = np.nansum((Y - Yh)[obs] ** 2)
        diff = abs(prev_ssr - ssr) / (abs(ssr) + 1e-10)
        prev_ssr = ssr
        F, Lambda = F_new, L_new

        if diff < tol:
            if verbose:
                print(f"    Converged at iter {it+1}, ssr={ssr:.4f}")
            break

    fpart = Lambda @ F.T
    if has_X:
        Xb = np.zeros((N, T))
        for k in range(p):
            Xb += beta[k] * X[:, :, k]
        Yh = Xb + fpart
    else:
        Yh = fpart
        beta = None

    return beta, F, Lambda, Yh


def cv_select_r(Y, X, r_max=6, n_folds=3):
    """Cross-validate number of factors."""
    N, T = Y.shape
    r_max = min(r_max, min(N, T) - 1)
    obs = ~np.isnan(Y)
    errs = np.full(r_max, np.inf)
    for r in range(1, r_max + 1):
        cv = []
        for _ in range(n_folds):
            mask = np.random.rand(N, T) < 0.2
            hold = mask & obs
            Y_tr = np.where(hold, np.nan, Y)
            try:
                _, _, _, Yh = estimate_ife(Y_tr, X, r, max_iter=50)
                cv.append(np.nanmean((Y[hold] - Yh[hold]) ** 2))
            except Exception:
                pass
        if cv:
            errs[r - 1] = np.mean(cv)
    return int(np.argmin(errs) + 1), errs


def gsc_estimate(Y, Tmat, X=None, r=None, r_max=6, verbose=True):
    """
    GSC (Xu 2017): IFE on controls -> loadings from pre-treatment -> counterfactual.
    Returns: att, att_time, Y_cf, att_unit, r_used
    """
    N, T = Y.shape
    treated = Tmat.any(axis=1)
    control = ~treated
    if control.sum() < 2:
        raise ValueError("Need >=2 controls")

    if r is None:
        r, _ = cv_select_r(Y[control], X[control] if X is not None else None, r_max)
        if verbose:
            print(f"  CV: r={r}")

    # IFE on controls
    beta, F, L_ctrl, Yh_ctrl = estimate_ife(Y[control],
                                             X[control] if X is not None else None, r=r)

    Y_cf = np.full((N, T), np.nan)
    Y_cf[control] = Yh_ctrl
    att_unit = np.full(N, np.nan)
    att_time = np.full(T, np.nan)

    for i in range(N):
        if not treated[i]:
            continue
        pre = (~Tmat[i]) & (~np.isnan(Y[i]))
        post = Tmat[i] & (~np.isnan(Y[i]))
        if pre.sum() < r:
            continue

        if X is not None and beta is not None:
            Xb_i = np.zeros(T)
            for k in range(len(beta)):
                Xb_i += beta[k] * X[i, :, k]
        else:
            Xb_i = np.zeros(T)

        y_pre_res = Y[i, pre] - Xb_i[pre]
        try:
            lam_i = np.linalg.lstsq(F[pre], y_pre_res, rcond=None)[0]
        except np.linalg.LinAlgError:
            lam_i = np.zeros(r)

        Y_cf[i] = Xb_i + F @ lam_i
        if post.sum() > 0:
            att_unit[i] = np.nanmean(Y[i, post] - Y_cf[i, post])

    for t in range(T):
        m = Tmat[:, t] & treated & (~np.isnan(Y[:, t])) & (~np.isnan(Y_cf[:, t]))
        if m.sum() > 0:
            att_time[t] = np.nanmean(Y[m, t] - Y_cf[m, t])

    return np.nanmean(att_time), att_time, Y_cf, att_unit, r


def event_study(Y, tyears, Y_cf, tids, pre_win=10, post_win=15):
    """Align treatment effects by event time (years relative to adoption)."""
    N, T = Y.shape
    y2i = {int(y): j for j, y in enumerate(tids)}
    gaps = {}

    for i in range(N):
        ty = tyears[i]
        if np.isnan(ty):
            continue
        t0 = y2i.get(int(ty))
        if t0 is None:
            continue
        start, end = max(0, t0 - pre_win), min(T, t0 + post_win + 1)
        for t in range(start, end):
            et = t - t0
            if not np.isnan(Y[i, t]) and not np.isnan(Y_cf[i, t]):
                g = float(Y[i, t] - Y_cf[i, t])
                if np.isfinite(g):
                    gaps.setdefault(et, []).append(g)

    et_vals = sorted(gaps.keys())
    att = np.array([np.nanmean(gaps[e]) for e in et_vals])
    se = np.array([np.nanstd(gaps[e]) / np.sqrt(max(len(gaps[e]), 1)) for e in et_vals])
    n_units = np.array([len(gaps[e]) for e in et_vals])
    return {
        "event_time": np.array(et_vals), "att": att, "se": se,
        "ci_lower": att - 1.96 * se, "ci_upper": att + 1.96 * se, "n_units": n_units
    }


def placebo_test(Y, Tmat, X, r, n_placebos=200):
    """Random reassignment of treatment to control units."""
    treated = Tmat.any(axis=1)
    ctrl_idx = np.where(~treated)[0]
    n_tr = treated.sum()
    if len(ctrl_idx) < n_tr:
        return np.array([])
    atts = []
    for _ in range(n_placebos):
        p_idx = np.random.choice(ctrl_idx, size=n_tr, replace=False)
        pm = np.zeros_like(Tmat)
        for i in p_idx:
            t0 = np.random.randint(Tmat.shape[1] // 4, 3 * Tmat.shape[1] // 4)
            pm[i, t0:] = True
        try:
            a, _, _, _, _ = gsc_estimate(Y, pm, X, r=r, verbose=False)
            if not np.isnan(a):
                atts.append(a)
        except Exception:
            pass
    return np.array(atts)


def prepare_data(panel, outcome_var, covariate_vars=None, unit_var="iso3",
                  time_var="year", treat_var="uhc_year"):
    """Convert panel DataFrame to GSC matrix format."""
    if covariate_vars is None:
        covariate_vars = []
    uids = sorted(panel[unit_var].unique())
    tids = sorted(panel[time_var].unique())
    N, T = len(uids), len(tids)
    u2i = {u: i for i, u in enumerate(uids)}
    t2i = {t: j for j, t in enumerate(tids)}

    Y = np.full((N, T), np.nan)
    for _, r in panel.iterrows():
        i, j = u2i.get(r[unit_var]), t2i.get(r[time_var])
        if i is not None and j is not None and pd.notna(r.get(outcome_var, np.nan)):
            Y[i, j] = r[outcome_var]

    Tmat = np.zeros((N, T), dtype=bool)
    tyears = np.full(N, np.nan)
    ut = panel.groupby(unit_var)[treat_var].first()
    for u in uids:
        i = u2i[u]
        if u in ut.index and pd.notna(ut[u]):
            tyears[i] = float(ut[u])
            for j, t in enumerate(tids):
                if t >= ut[u]:
                    Tmat[i, j] = True

    X = None
    if covariate_vars:
        cvars = [c for c in covariate_vars if c in panel.columns]
        if cvars:
            p = len(cvars)
            X = np.full((N, T, p), np.nan)
            for k, var in enumerate(cvars):
                for _, r in panel.iterrows():
                    i, j = u2i.get(r[unit_var]), t2i.get(r[time_var])
                    if i is not None and j is not None and pd.notna(r[var]):
                        X[i, j, k] = r[var]
            for k in range(p):
                m = np.nanmean(X[:, :, k])
                X[:, :, k] = np.where(np.isnan(X[:, :, k]), m, X[:, :, k])

    return Y, Tmat, X, uids, tids, tyears, u2i, t2i


def filter_panel(panel):
    """Apply all exclusions and window restrictions."""
    p = panel.copy()
    p = p[(p["year"] >= YEAR_START) & (p["year"] <= YEAR_END)]
    p = p[~p["iso3"].isin(EXCLUDE_NO_DATA)]
    # Nullify treatment for excluded countries (must do before modifying "treated")
    p.loc[p["iso3"].isin(EXCLUDE_TREATED), "uhc_year"] = np.nan
    p.loc[p["iso3"].isin(EXCLUDE_TREATED), "treated"] = 0
    return p


def load_panel():
    return pd.read_csv(PROC_DIR / "analysis_panel.csv")


def get_covariates():
    return [c for c in COVARIATES if c in load_panel().columns]
