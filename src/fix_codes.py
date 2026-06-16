"""
Step 2: Fix country codes (ISO2->ISO3) and build country classification.
"""
import pandas as pd
import numpy as np
import pycountry
from config import RAW_DIR


def build_iso_map():
    m = {}
    for c in pycountry.countries:
        if hasattr(c, "alpha_2") and hasattr(c, "alpha_3"):
            m[c.alpha_2] = c.alpha_3
    m.update({"XK": "XKX", "AN": "ANT", "CS": "SCG"})
    return m


def fix_worldbank():
    """Convert WB ISO2 to ISO3, filter non-countries."""
    iso_map = build_iso_map()
    wb = pd.read_csv(RAW_DIR / "worldbank_wdi.csv", low_memory=False)

    # Keep only rows with 2-letter ISO codes
    wb = wb[wb["country_code"].str.match(r"^[A-Z]{2}$", na=False)].copy()
    wb["iso3"] = wb["country_code"].map(iso_map)
    n_fail = wb["iso3"].isna().sum()
    if n_fail:
        failed = wb[wb["iso3"].isna()]["country_code"].unique()
        print(f"  Unmapped codes: {list(failed)}")
    wb = wb.dropna(subset=["iso3"])

    # Pivot to wide
    val_cols = [c for c in wb.columns if c not in ("country_code", "country_name", "iso3", "year")]
    wb_long = wb.melt(id_vars=["iso3", "year"], value_vars=val_cols,
                       var_name="indicator", value_name="value").dropna(subset=["value"])
    wb_wide = wb_long.pivot_table(index=["iso3", "year"], columns="indicator",
                                   values="value").reset_index()
    wb_wide["year"] = wb_wide["year"].astype(int)

    fname = RAW_DIR / "worldbank_clean.csv"
    wb_wide.to_csv(fname, index=False)
    print(f"  World Bank clean: {len(wb_wide)} rows, {wb_wide['iso3'].nunique()} countries -> {fname}")
    return wb_wide


def build_region_map():
    """Manual region classification for all countries."""
    SOUTH_ASIA = {"AFG", "BGD", "BTN", "IND", "LKA", "MDV", "NPL", "PAK"}
    EAST_ASIA_PAC = {"AUS", "BRN", "CHN", "FJI", "IDN", "JPN", "KHM", "KOR", "LAO",
                      "MMR", "MNG", "MYS", "NZL", "PHL", "PNG", "PRK", "SGP", "SLB",
                      "THA", "TLS", "VNM", "VUT", "WSM", "KIR", "TON", "FSM", "MHL",
                      "PLW", "NRU", "HKG", "MAC", "TWN"}
    EUROPE_CENTRAL = {"ALB", "AND", "AUT", "BEL", "BGR", "BIH", "BLR", "CHE", "CYP",
                       "CZE", "DEU", "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "GRC",
                       "HRV", "HUN", "IRL", "ISL", "ITA", "LTU", "LUX", "LVA", "MCO",
                       "MDA", "MKD", "MLT", "MNE", "NLD", "NOR", "POL", "PRT", "ROU",
                       "SRB", "SVK", "SVN", "SWE", "UKR", "XKX", "ARM", "AZE", "GEO",
                       "KAZ", "KGZ", "RUS", "TJK", "TKM", "TUR", "UZB"}
    LATAM = {"ARG", "BOL", "BRA", "CHL", "COL", "CRI", "CUB", "DOM", "ECU", "GTM",
             "HND", "HTI", "JAM", "MEX", "NIC", "PAN", "PER", "PRY", "SLV", "URY",
             "VEN", "BLZ", "GUY", "SUR", "BHS", "BRB", "TTO", "LCA", "VCT", "GRD",
             "ATG", "DMA", "KNA"}
    MENA = {"DZA", "BHR", "DJI", "EGY", "IRN", "IRQ", "ISR", "JOR", "KWT", "LBN",
            "LBY", "MAR", "OMN", "QAT", "SAU", "SYR", "TUN", "ARE", "YEM", "PSE"}
    SUB_SAHARA = set()  # fill by exclusion
    NORTH_AM = {"USA", "CAN"}

    region_map = {}
    for iso3 in SOUTH_ASIA: region_map[iso3] = "South Asia"
    for iso3 in EAST_ASIA_PAC: region_map[iso3] = "East Asia & Pacific"
    for iso3 in EUROPE_CENTRAL: region_map[iso3] = "Europe & Central Asia"
    for iso3 in LATAM: region_map[iso3] = "Latin America & Caribbean"
    for iso3 in MENA: region_map[iso3] = "Middle East & North Africa"
    for iso3 in NORTH_AM: region_map[iso3] = "North America"

    # Remaining by continent via pycountry
    for c in pycountry.countries:
        if not hasattr(c, "alpha_3"):
            continue
        iso3 = c.alpha_3
        if iso3 in region_map:
            continue
        cont = getattr(c, "continent", "")
        if cont == "AF":
            region_map[iso3] = "Sub-Saharan Africa"
        elif cont == "AS":
            region_map[iso3] = "East Asia & Pacific"
        elif cont == "EU":
            region_map[iso3] = "Europe & Central Asia"
        elif cont == "NA":
            region_map[iso3] = "North America"
        elif cont == "SA":
            region_map[iso3] = "Latin America & Caribbean"
        elif cont == "OC":
            region_map[iso3] = "East Asia & Pacific"
        else:
            region_map[iso3] = "Other"
    return region_map


def save_country_classification():
    region_map = build_region_map()
    records = [{"iso3": k, "region": v} for k, v in region_map.items()]
    df = pd.DataFrame(records)
    fname = RAW_DIR / "country_classification.csv"
    df.to_csv(fname, index=False)
    print(f"  Classification: {len(df)} countries -> {fname}")
    return df


def main():
    print("=" * 50)
    print("Fixing country codes")
    fix_worldbank()
    print("\n" + "=" * 50)
    print("Building country classification")
    save_country_classification()
    print("\nDone.")


if __name__ == "__main__":
    main()
