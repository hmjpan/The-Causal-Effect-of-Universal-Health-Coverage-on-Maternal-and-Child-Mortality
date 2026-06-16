"""
Step 3: Build analysis panel from raw data.
Merges WHO mortality, UHC index, World Bank covariates, and country metadata.
"""
import pandas as pd
import numpy as np
from config import RAW_DIR, PROC_DIR, EXCLUDE_TREATED, EXCLUDE_NO_DATA


def load_who(filepath, keep_dim2=None, keep_dim3=None):
    """Load & clean a WHO GHO indicator CSV."""
    df = pd.read_csv(filepath, low_memory=False)
    if "SpatialDimType" in df.columns:
        df = df[df["SpatialDimType"] == "COUNTRY"]
    if "TimeDimType" in df.columns:
        df = df[df["TimeDimType"] == "YEAR"]

    # Sex: both sexes
    if "Dim1" in df.columns and df["Dim1"].nunique() > 1:
        if "SEX_BTSX" in df["Dim1"].values:
            df = df[df["Dim1"] == "SEX_BTSX"]

    # Dim2 / Dim3 filters
    for dim_col, keep_vals in [("Dim2", keep_dim2), ("Dim3", keep_dim3)]:
        if keep_vals and dim_col in df.columns:
            df = df[df[dim_col].isin(keep_vals)]
        elif dim_col in df.columns and df[dim_col].notna().any():
            totals = [v for v in df[dim_col].dropna().unique()
                       if isinstance(v, str) and "_TOTL" in v.upper()]
            if totals:
                df = df[df[dim_col].isin(totals)]

    df["year"] = pd.to_numeric(df["TimeDim"], errors="coerce")
    df["iso3"] = df["SpatialDim"].astype(str)
    val_col = "NumericValue" if "NumericValue" in df.columns else "Value"
    df["value"] = pd.to_numeric(df[val_col], errors="coerce")
    return df[["iso3", "year", "value"]].dropna()


def build_panel():
    print("=" * 50)
    print("Building analysis panel")

    # ---- Load outcomes ----
    ncd = load_who(RAW_DIR / "who_ncd_mortality.csv", keep_dim2=["AGEGROUP_YEARS30-69"])
    ncd = ncd.rename(columns={"value": "ncd_mortality_3070"})

    u5 = load_who(RAW_DIR / "who_under5_mortality.csv",
                   keep_dim2=["AGEGROUP_YEARSUNDER5"], keep_dim3=["WEALTHQUINTILE_TOTL"])
    u5 = u5.rename(columns={"value": "under5_mortality"})

    mat = load_who(RAW_DIR / "who_maternal_mortality.csv",
                    keep_dim2=["AGEGROUP_MONTHS0-11"])
    mat = mat.rename(columns={"value": "maternal_mortality"})

    le = load_who(RAW_DIR / "who_life_expectancy.csv")
    le = le.rename(columns={"value": "life_expectancy"})

    uhc = load_who(RAW_DIR / "who_uhc_index.csv")
    uhc = uhc.rename(columns={"value": "uhc_index"})

    dpt3 = load_who(RAW_DIR / "who_dpt3_immun.csv")
    dpt3 = dpt3.rename(columns={"value": "dpt3_coverage"})

    measles = load_who(RAW_DIR / "who_measles_immun.csv")
    measles = measles.rename(columns={"value": "measles_coverage"})

    beds = load_who(RAW_DIR / "who_hospital_beds.csv")
    beds = beds.rename(columns={"value": "hospital_beds"})

    print(f"  Outcomes loaded: NCD={len(ncd)}, U5={len(u5)}, Mat={len(mat)}, "
          f"LE={len(le)}, UHC={len(uhc)}")
    print(f"  Mechanisms: DPT3={len(dpt3)}, Measles={len(measles)}, Beds={len(beds)}")

    # ---- Load World Bank ----
    wb = pd.read_csv(RAW_DIR / "worldbank_clean.csv")
    wb["year"] = wb["year"].astype(int)
    print(f"  World Bank: {len(wb)} obs, {wb['iso3'].nunique()} countries")

    # ---- Load classifications and timeline ----
    clf = pd.read_csv(RAW_DIR / "country_classification.csv")
    timeline = pd.read_csv(RAW_DIR / "uhc_timeline.csv")

    # ---- Merge all ----
    panel = uhc.merge(ncd, on=["iso3", "year"], how="left")
    for df in [u5, mat, le, dpt3, measles, beds]:
        panel = panel.merge(df, on=["iso3", "year"], how="left")
    panel = panel.merge(wb, on=["iso3", "year"], how="left")
    panel = panel.merge(clf, on="iso3", how="left")
    panel = panel.merge(timeline[["iso3", "uhc_year", "uhc_type"]], on="iso3", how="left")

    panel["treated"] = panel["uhc_year"].notna().astype(int)

    # ---- Derived variables ----
    for col in ["ncd_mortality_3070", "under5_mortality", "maternal_mortality",
                "gdp_per_capita_ppp", "population", "health_expenditure_per_capita"]:
        if col in panel.columns:
            panel[f"log_{col}"] = np.log(panel[col].clip(lower=0) + 1)

    panel["uhc_change"] = panel.groupby("iso3")["uhc_index"].diff()
    panel["post_treatment"] = (panel["year"] >= panel["uhc_year"]).astype(int)
    panel.loc[panel["uhc_year"].isna(), "post_treatment"] = 0

    # ---- Filter ----
    panel = panel[panel.groupby("iso3")["year"].transform("nunique") >= 5]
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)

    # Remove aggregate rows
    aggs = {"GLOBAL", "AFR", "AMR", "EMR", "EUR", "SEAR", "WPR", "OWID_WRL",
            "WB_LI", "WB_LMI", "WB_UMI", "WB_HI", "SAS", "ECS", "MEA", "NAC",
            "LCN", "EAS", "SSF"}
    panel = panel[~panel["iso3"].isin(aggs)]

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    fname = PROC_DIR / "analysis_panel.csv"
    panel.to_csv(fname, index=False)

    print(f"\n  Panel: {len(panel)} obs, {panel['iso3'].nunique()} countries, "
          f"{panel['year'].min()}-{panel['year'].max()}")
    print(f"  Treated: {panel[panel['treated']==1]['iso3'].nunique()}")
    print(f"  Control: {panel[panel['treated']==0]['iso3'].nunique()}")
    print(f"  Saved: {fname}")
    return panel


if __name__ == "__main__":
    build_panel()
