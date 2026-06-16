"""
Step 6: All visualizations and tables. Imports from config, no hardcoded paths.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from config import FIG_DIR, TBL_DIR, PROC_DIR, EXCLUDE_TREATED, EXCLUDE_NO_DATA

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
})


def load_results():
    with open(PROC_DIR / "gsc_results.pkl", "rb") as f:
        return pickle.load(f)


# ---- Figure 1: ATT Bar Chart ----
def fig_att_barchart(res):
    main = res["main"]
    labels_map = {
        "log_under5_mortality": "Under-5\nMortality",
        "log_maternal_mortality": "Maternal\nMortality",
        "log_ncd_mortality_3070": "NCD Mortality\n(30-70)",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    atts, lbls, pvs, cols = [], [], [], []
    for k in ["log_under5_mortality", "log_maternal_mortality", "log_ncd_mortality_3070"]:
        if k in main:
            atts.append(main[k]["att"])
            lbls.append(labels_map[k])
            pvs.append(main[k].get("p_value", np.nan))
            cols.append("#4CAF50" if main[k]["att"] < 0 else "#F44336")
    x = np.arange(len(atts))
    ax.bar(x, atts, color=cols, width=0.55, edgecolor="white")
    for i, (att, pv) in enumerate(zip(atts, pvs)):
        sig = "***" if pv < 0.01 else ("**" if pv < 0.05 else "n.s.")
        ax.text(i, att - 0.005 if att < 0 else att + 0.003, f"{att:.3f} {sig}",
                ha="center", va="top" if att < 0 else "bottom", fontweight="bold", fontsize=11)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(lbls)
    ax.set_ylabel("ATT (log points)", fontweight="bold")
    ax.set_title("Causal Effect of UHC on Mortality (GSC, r=1)", fontweight="bold")
    ax.annotate("*** p<0.01  ** p<0.05  n.s. not significant",
                xy=(0.5, 0.01), xycoords="axes fraction", ha="center", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_att.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig1_att.png")


# ---- Figure 2: Event Study ----
def fig_event_study(res):
    main = res["main"]
    outcomes = {
        "log_under5_mortality": ("Under-5 Mortality", "#4CAF50"),
        "log_maternal_mortality": ("Maternal Mortality", "#FF9800"),
        "log_ncd_mortality_3070": ("NCD Mortality", "#2196F3"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (k, (lb, c)) in enumerate(outcomes.items()):
        if k not in main:
            continue
        ax = axes[idx]
        ev = main[k]["event_study"]
        et = np.array(ev["event_time"]); att = np.array(ev["att"])
        ci_l = np.array(ev["ci_lower"]); ci_u = np.array(ev["ci_upper"])
        valid = ~np.isnan(att)
        if valid.sum() == 0:
            ax.set_title(lb); continue
        ax.fill_between(et[valid], ci_l[valid], ci_u[valid], alpha=0.15, color=c)
        ax.plot(et[valid], att[valid], "o-", color=c, linewidth=1.8, markersize=4)
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        ax.axvline(x=-0.5, color="red", linewidth=1, linestyle=":", label="UHC")
        pre_m = np.nanmean(att[et < 0]) if (et < 0).any() else np.nan
        post_m = np.nanmean(att[et >= 0]) if (et >= 0).any() else np.nan
        ax.set_title(f"{lb}\npre={pre_m:.3f}, post={post_m:.3f}", fontweight="bold", fontsize=10)
        ax.set_xlabel("Years relative to UHC adoption")
        ax.set_ylabel("ATT (log points)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Event Study: Mortality Before and After UHC Adoption", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_event_study.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig2_event_study.png")


# ---- Figure 3: Placebo ----
def fig_placebo(res):
    main = res["main"]
    outcomes = {
        "log_under5_mortality": "Under-5 Mortality",
        "log_maternal_mortality": "Maternal Mortality",
        "log_ncd_mortality_3070": "NCD Mortality",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (k, lb) in enumerate(outcomes.items()):
        if k not in main:
            continue
        ax = axes[idx]
        patts = np.array(main[k].get("placebo_atts", []))
        att = main[k]["att"]
        pv = main[k].get("p_value", np.nan)
        if len(patts) == 0:
            ax.set_title(lb); continue
        ax.hist(patts, bins=15, color="gray", alpha=0.4, edgecolor="white", density=True)
        ax.axvline(x=att, color="red", linewidth=2, linestyle="--", label=f"ATT={att:.3f}")
        ax.axvline(x=0, color="black", linewidth=0.5)
        sig = "Significant" if pv < 0.05 else "Not significant"
        ax.set_title(f"{lb}\nPlacebo p={pv:.3f} ({sig})", fontweight="bold", fontsize=10)
        ax.set_xlabel("ATT"); ax.set_ylabel("Density"); ax.legend(fontsize=8)
    fig.suptitle("Placebo Test: Actual ATT vs Random Assignment (n=200)", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_placebo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig3_placebo.png")


# ---- Figure 4: Trends ----
def fig_trends():
    panel = pd.read_csv(PROC_DIR / "analysis_panel.csv")
    panel = panel[(panel["year"] >= 2000) & (panel["year"] <= 2021)]
    panel = panel[~panel["iso3"].isin(EXCLUDE_NO_DATA)]
    panel.loc[(panel["treated"] == 1) & (panel["iso3"].isin(EXCLUDE_TREATED)), "treated"] = 0
    panel["grp"] = "Control"
    panel.loc[panel["treated"] == 1, "grp"] = "Treated"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (var, title) in enumerate([
        ("uhc_index", "UHC Index"), ("ncd_mortality_3070", "NCD Mortality (30-70)"),
        ("under5_mortality", "Under-5 Mortality"), ("maternal_mortality", "Maternal Mortality")
    ]):
        ax = axes[idx // 2, idx % 2]
        for grp, c in [("Treated", "#F44336"), ("Control", "#2196F3")]:
            s = panel[panel["grp"] == grp].groupby("year")[var].mean()
            ax.plot(s.index, s.values, "-", linewidth=2, label=grp, color=c)
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("Year"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.suptitle("Trends: Treated vs Control Countries", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_trends.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig4_trends.png")


# ---- Tables ----
def table_main_results(res):
    main = res["main"]
    pre = res.get("pre_covid", {})
    rows = []
    for k in ["log_under5_mortality", "log_maternal_mortality", "log_ncd_mortality_3070"]:
        m = main.get(k, {})
        rows.append({
            "Outcome": k.replace("log_", "").replace("_", " ").title(),
            "ATT (GSC r=1)": f"{m.get('att', np.nan):.4f}",
            "Placebo p": f"{m.get('p_value', np.nan):.4f}",
            "Pre-COVID ATT": f"{pre.get(k, np.nan):.4f}",
        })
    df = pd.DataFrame(rows)
    fname = TBL_DIR / "table1_main.csv"
    df.to_csv(fname, index=False)
    print(f"  {fname}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating figures and tables...")
    res = load_results()
    fig_att_barchart(res)
    fig_event_study(res)
    fig_placebo(res)
    fig_trends()
    table_main_results(res)
    print("Done.")


if __name__ == "__main__":
    main()
