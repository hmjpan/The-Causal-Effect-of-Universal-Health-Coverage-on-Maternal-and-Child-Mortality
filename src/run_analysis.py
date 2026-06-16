"""
Step 5: Main GSC analysis + all robustness checks.
Imports gsc_core — no hardcoded paths.
"""
import numpy as np
import pandas as pd
import pickle
from config import PROC_DIR, FIG_DIR, TBL_DIR, COVARIATES
from config import EXCLUDE_TREATED, EXCLUDE_NO_DATA, YEAR_START, YEAR_END
from gsc_core import (
    estimate_ife, gsc_estimate, event_study, placebo_test,
    prepare_data, filter_panel, load_panel, get_covariates
)


def run_gsc_main(panel, y_var, label, covariates, r=1):
    """Run GSC + event study + placebo for one outcome."""
    Y, Tmat, X, uids, tids, tyears, _, _ = prepare_data(panel, y_var, covariate_vars=covariates)
    n_tr = Tmat.any(axis=1).sum()
    n_co = (~Tmat.any(axis=1)).sum()
    print(f"  {n_tr} treated, {n_co} control, {len(tids)} periods")

    att, att_time, Y_cf, att_unit, r_used = gsc_estimate(Y, Tmat, X, r=r, verbose=True)
    ev = event_study(Y, tyears, Y_cf, tids)
    pre_att = np.nanmean(ev["att"][ev["event_time"] < 0]) if (ev["event_time"] < 0).any() else np.nan
    post_att = np.nanmean(ev["att"][ev["event_time"] >= 0]) if (ev["event_time"] >= 0).any() else np.nan
    p_atts = placebo_test(Y, Tmat, X, r_used, n_placebos=200)
    p_val = np.mean(np.abs(p_atts) >= np.abs(att)) if len(p_atts) > 0 and not np.isnan(att) and att != 0 else np.nan

    pct = (1 - np.exp(att)) * 100
    print(f"  ATT={att:.4f} ({abs(pct):.1f}%), pre-trend={pre_att:.4f}, post-event={post_att:.4f}, p={p_val:.4f}")

    return {
        "att": att, "att_time": att_time, "att_unit": att_unit,
        "event_study": ev, "placebo_atts": p_atts, "p_value": p_val,
        "r_used": r_used, "uids": uids, "tids": tids, "tyears": tyears,
    }


def run_all_outcomes(panel, covariates):
    outcomes = {
        "log_under5_mortality": "Under-5 Mortality",
        "log_maternal_mortality": "Maternal Mortality",
        "log_ncd_mortality_3070": "NCD Mortality",
    }
    results = {}
    for y_var, label in outcomes.items():
        if y_var not in panel.columns:
            continue
        print(f"\n{'='*50}\n{label}\n{'='*50}")
        results[y_var] = run_gsc_main(panel, y_var, label, covariates, r=1)
    return results


def run_robustness_placebo(results):
    """Return clean placebo summary."""
    return {k: {"p_val": v["p_value"], "n_placebos": len(v["placebo_atts"])}
            for k, v in results.items()}


def run_factor_diagnostics(panel, y_var):
    """Eigenvalue ratio test for number of factors."""
    Y, _, _, _, tids, _, _, _ = prepare_data(panel, y_var, covariate_vars=None)
    N, T = Y.shape
    Y_fill = np.where(~np.isnan(Y), Y, np.nanmean(Y))
    Y_dm = Y_fill - np.nanmean(Y_fill, axis=0)
    corr = (Y_dm.T @ Y_dm) / N
    w, _ = np.linalg.eigh(corr)
    w = w[::-1]
    er = []
    for i in range(min(8, len(w) - 1)):
        er.append(w[i] / w[i + 1] if w[i + 1] > 1e-10 else 1.0)
    r_er = int(np.argmax(er) + 1)
    var_exp = w[0] / np.sum(w)
    return {"r_er": r_er, "er_ratios": er, "top_ev": w[:5], "var_exp_r1": var_exp}


def run_region_matched_gsc(panel, y_var, covariates):
    """GSC with donors restricted to same region."""
    clf = pd.read_csv(PROC_DIR.parent / "raw" / "country_classification.csv")
    region_map = dict(zip(clf["iso3"], clf["region"]))
    Y, Tmat, X, uids, _, _, _, _ = prepare_data(panel, y_var, covariate_vars=covariates)
    gaps_all = []
    for i in np.where(Tmat.any(axis=1))[0]:
        ri = region_map.get(uids[i], "Other")
        same_reg = np.array([j for j in range(len(uids))
                             if not Tmat[j].any() and region_map.get(uids[j], "") == ri])
        if len(same_reg) < 5:
            same_reg = np.array([j for j in range(len(uids)) if not Tmat[j].any()])
        keep = np.zeros(len(uids), dtype=bool)
        keep[i] = True
        keep[same_reg] = True
        try:
            _, _, Ycf, _, _ = gsc_estimate(Y[keep], Tmat[keep],
                                            X[keep] if X is not None else None, r=1, verbose=False)
            ti = np.where(keep)[0].tolist().index(i)
            post = Tmat[i]
            gaps_all.append(np.nanmean(Y[i, post] - Ycf[ti, post]))
        except Exception:
            pass
    return np.nanmean(gaps_all) if gaps_all else np.nan


def run_in_time_placebo(panel, y_var, covariates, shift=5):
    """Backdate pseudo-treatment by shift years."""
    Y, Tmat, X, uids, tids, tyears, u2i, t2i = prepare_data(panel, y_var, covariate_vars=covariates)
    Tmat_fake = np.zeros_like(Tmat)
    for i in range(len(uids)):
        ty = tyears[i]
        if np.isnan(ty):
            continue
        ti = t2i.get(int(ty))
        if ti is None or ti - shift < 2:
            continue
        Tmat_fake[i, max(0, ti - shift):ti] = True
    if Tmat_fake.sum() == 0:
        return np.nan
    a, _, _, _, _ = gsc_estimate(Y, Tmat_fake, X, r=1, verbose=False)
    return a


def main():
    panel = load_panel()
    panel = filter_panel(panel)
    covariates = get_covariates()
    n_treated = panel[panel["treated"] == 1]["iso3"].nunique()
    n_control = panel[panel["treated"] == 0]["iso3"].nunique()
    print(f"Panel: {len(panel)} obs, {n_treated + n_control} countries ({n_treated} treated, {n_control} control)")

    # ---- 1. Main GSC ----
    print("\n" + "#" * 60 + "\n# MAIN GSC ANALYSIS\n" + "#" * 60)
    gsc_results = run_all_outcomes(panel, covariates)

    # ---- 2. Factor diagnostics ----
    print("\n" + "#" * 60 + "\n# FACTOR DIAGNOSTICS\n" + "#" * 60)
    for y_var in gsc_results:
        fd = run_factor_diagnostics(panel, y_var)
        print(f"  {y_var}: r_ER={fd['r_er']}, ER1={fd['er_ratios'][0]:.1f}, var_exp(r=1)={fd['var_exp_r1']:.1%}")

    # ---- 3. Pre-COVID window ----
    print("\n" + "#" * 60 + "\n# PRE-COVID WINDOW (2000-2019)\n" + "#" * 60)
    panel_pre = panel[panel["year"] <= 2019].copy()
    pre_results = run_all_outcomes(panel_pre, covariates)

    # ---- 4. Region-matched ----
    print("\n" + "#" * 60 + "\n# REGION-MATCHED GSC\n" + "#" * 60)
    for y_var in gsc_results:
        rm = run_region_matched_gsc(panel, y_var, covariates)
        print(f"  {y_var}: ATT_rm = {rm:.4f}")

    # ---- 5. In-time placebo ----
    print("\n" + "#" * 60 + "\n# IN-TIME PLACEBO\n" + "#" * 60)
    for y_var in gsc_results:
        itp = run_in_time_placebo(panel, y_var, covariates, shift=5)
        print(f"  {y_var}: ATT_itp(5yr) = {itp:.4f}")

    # ---- Save ----
    all_res = {
        "main": {k: {kk: vv for kk, vv in v.items() if kk != "Y_cf"} for k, v in gsc_results.items()},
        "pre_covid": {k: v["att"] for k, v in pre_results.items()},
        "region_matched": {k: run_region_matched_gsc(panel, k, covariates) for k in gsc_results},
    }
    # Convert arrays to lists for pickle
    for k, v in all_res["main"].items():
        v["att_time"] = v["att_time"].tolist()
        v["att_unit"] = v["att_unit"].tolist()
        v["event_study"] = {kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                            for kk, vv in v["event_study"].items()}
        v["placebo_atts"] = v["placebo_atts"].tolist()

    with open(PROC_DIR / "gsc_results.pkl", "wb") as f:
        pickle.dump(all_res, f)
    print(f"\nResults saved to {PROC_DIR / 'gsc_results.pkl'}")

    return all_res


if __name__ == "__main__":
    main()
