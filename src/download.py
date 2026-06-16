"""
Step 1: Download all raw data from WHO GHO and World Bank APIs.
All data is free, publicly available, no registration required.
"""
import pandas as pd
import numpy as np
import urllib.request
import json
import time
from pathlib import Path
from config import RAW_DIR, EXCLUDE_TREATED


def fetch_wb_indicator(code, date_range="2000:2024"):
    """Download one World Bank WDI indicator."""
    url = f"http://api.worldbank.org/v2/country/all/indicator/{code}?format=json&per_page=20000&date={date_range}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data[1] if len(data) >= 2 and data[1] else None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def download_worldbank():
    """Download all WB indicators and save."""
    WB = {
        "NY.GDP.PCAP.PP.KD": "gdp_per_capita_ppp",
        "NY.GDP.MKTP.KD.ZG": "gdp_growth",
        "SP.POP.TOTL": "population",
        "SP.URB.TOTL.IN.ZS": "urban_population_pct",
        "SP.DYN.LE00.IN": "life_expectancy_total",
        "SP.DYN.CDRT.IN": "death_rate",
        "SH.XPD.CHEX.GD.ZS": "health_expenditure_pct_gdp",
        "SH.XPD.CHEX.PP.CD": "health_expenditure_per_capita",
        "SH.XPD.OOPC.CH.ZS": "oop_health_expenditure_pct",
        "SH.MED.PHYS.ZS": "physicians_per_1000",
        "SH.MED.NUMW.P3": "nurses_midwives_per_1000",
        "SH.STA.BRTC.ZS": "births_attended_skilled",
        "SP.POP.65UP.TO.ZS": "population_65_plus",
        "SI.POV.GINI": "gini_index",
        "SE.ADT.LITR.ZS": "literacy_rate",
        "EN.ATM.PM25.MC.M3": "pm25_exposure",
        "SP.DYN.IMRT.IN": "infant_mortality",
        "SP.DYN.TFRT.IN": "fertility_rate",
        "IT.NET.USER.ZS": "internet_users_pct",
        "EG.ELC.ACCS.ZS": "electricity_access_pct",
        "SH.H2O.BASW.ZS": "basic_water_access",
        "SH.STA.BASS.ZS": "basic_sanitation_access",
        "AG.PRD.FOOD.XD": "food_production_index",
        "NV.AGR.TOTL.ZS": "agriculture_gdp_pct",
        "EN.POP.DNST": "population_density",
        "NE.EXP.GNFS.ZS": "exports_pct_gdp",
        "GC.TAX.TOTL.GD.ZS": "tax_revenue_pct_gdp",
        "SL.UEM.TOTL.ZS": "unemployment_pct",
    }

    print("=" * 50)
    print("Downloading World Bank WDI")
    all_records = []
    for code, name in WB.items():
        print(f"  {name} ...", end=" ", flush=True)
        items = fetch_wb_indicator(code)
        if items:
            for item in items:
                if item.get("value") is not None:
                    all_records.append({
                        "country_code": item["country"]["id"],
                        "country_name": item["country"]["value"],
                        "indicator_code": code,
                        "variable": name,
                        "year": int(item["date"]),
                        "value": float(item["value"]),
                    })
            print(f"{len(items)} records")
        else:
            print("NO DATA")
        time.sleep(0.3)

    df = pd.DataFrame(all_records)
    df_wide = df.pivot_table(index=["country_code", "country_name", "year"],
                              columns="variable", values="value").reset_index()
    fname = RAW_DIR / "worldbank_wdi.csv"
    df_wide.to_csv(fname, index=False)
    print(f"  Saved: {fname} ({len(df_wide)} rows, {df_wide['country_code'].nunique()} codes)")
    return df_wide


def download_who_indicators():
    """Download key WHO GHO indicators."""
    import requests
    GHO = "https://ghoapi.azureedge.net/api"
    indicators = {
        "uhc_index": "UHC_INDEX_REPORTED",
        "ncd_mortality": "NCDMORT3070",
        "under5_mortality": "MDG_0000000007",
        "maternal_mortality": "MDG_0000000001",
        "life_expectancy": "WHOSIS_000001",
        "hale": "WHOSIS_000002",
        "dpt3_immun": "WHS4_543",
        "measles_immun": "WHS4_544",
        "tb_incidence": "MDG_0000000025",
        "hospital_beds": "HWF_0001",
        "alcohol": "SA_0000001400",
        "raised_bp": "BP_04",
    }

    print("\n" + "=" * 50)
    print("Downloading WHO GHO indicators")
    for name, code in indicators.items():
        fname = RAW_DIR / f"who_{name}.csv"
        if fname.exists():
            print(f"  {name}: already exists, skipping")
            continue
        try:
            r = requests.get(f"{GHO}/{code}", timeout=60)
            r.raise_for_status()
            data = r.json()
            records = data.get("value", [])
            if records:
                pd.DataFrame(records).to_csv(fname, index=False)
                print(f"  {name}: {len(records)} rows")
            else:
                print(f"  {name}: empty")
        except Exception as e:
            print(f"  {name}: ERROR {e}")
        time.sleep(1)


def save_uhc_timeline():
    """Save UHC adoption timeline (42 countries, literature-based)."""
    timeline = [
        {"iso3": "AFG", "uhc_year": 2019, "uhc_type": "tax_based"},
        {"iso3": "BFA", "uhc_year": 2016, "uhc_type": "tax_based"},
        {"iso3": "BGD", "uhc_year": 2011, "uhc_type": "insurance"},
        {"iso3": "BRA", "uhc_year": 1988, "uhc_type": "tax_based"},
        {"iso3": "CHL", "uhc_year": 2005, "uhc_type": "insurance"},
        {"iso3": "CHN", "uhc_year": 2009, "uhc_type": "insurance"},
        {"iso3": "COL", "uhc_year": 1993, "uhc_type": "insurance"},
        {"iso3": "EGY", "uhc_year": 2019, "uhc_type": "insurance"},
        {"iso3": "ETH", "uhc_year": 2011, "uhc_type": "community"},
        {"iso3": "GEO", "uhc_year": 2013, "uhc_type": "tax_based"},
        {"iso3": "GHA", "uhc_year": 2005, "uhc_type": "insurance"},
        {"iso3": "IDN", "uhc_year": 2014, "uhc_type": "insurance"},
        {"iso3": "IND", "uhc_year": 2018, "uhc_type": "insurance"},
        {"iso3": "IRN", "uhc_year": 2014, "uhc_type": "insurance"},
        {"iso3": "IRQ", "uhc_year": 2015, "uhc_type": "tax_based"},
        {"iso3": "JOR", "uhc_year": 2004, "uhc_type": "insurance"},
        {"iso3": "KEN", "uhc_year": 2018, "uhc_type": "insurance"},
        {"iso3": "KGZ", "uhc_year": 2001, "uhc_type": "insurance"},
        {"iso3": "KHM", "uhc_year": 1996, "uhc_type": "tax_based"},
        {"iso3": "LAO", "uhc_year": 2016, "uhc_type": "insurance"},
        {"iso3": "LKA", "uhc_year": 1950, "uhc_type": "tax_based"},
        {"iso3": "MAR", "uhc_year": 2012, "uhc_type": "insurance"},
        {"iso3": "MDA", "uhc_year": 2004, "uhc_type": "insurance"},
        {"iso3": "MEX", "uhc_year": 2004, "uhc_type": "insurance"},
        {"iso3": "MLI", "uhc_year": 2019, "uhc_type": "insurance"},
        {"iso3": "MMR", "uhc_year": 2015, "uhc_type": "tax_based"},
        {"iso3": "MNG", "uhc_year": 1994, "uhc_type": "insurance"},
        {"iso3": "MYS", "uhc_year": 1980, "uhc_type": "tax_based"},
        {"iso3": "NGA", "uhc_year": 2014, "uhc_type": "insurance"},
        {"iso3": "NPL", "uhc_year": 2008, "uhc_type": "tax_based"},
        {"iso3": "PAK", "uhc_year": 2015, "uhc_type": "insurance"},
        {"iso3": "PER", "uhc_year": 2009, "uhc_type": "insurance"},
        {"iso3": "PHL", "uhc_year": 1995, "uhc_type": "insurance"},
        {"iso3": "RWA", "uhc_year": 2005, "uhc_type": "community"},
        {"iso3": "SEN", "uhc_year": 2013, "uhc_type": "insurance"},
        {"iso3": "THA", "uhc_year": 2002, "uhc_type": "tax_based"},
        {"iso3": "TUN", "uhc_year": 2004, "uhc_type": "insurance"},
        {"iso3": "TUR", "uhc_year": 2003, "uhc_type": "insurance"},
        {"iso3": "UKR", "uhc_year": 2018, "uhc_type": "tax_based"},
        {"iso3": "UZB", "uhc_year": 2020, "uhc_type": "tax_based"},
        {"iso3": "VNM", "uhc_year": 1993, "uhc_type": "insurance"},
        {"iso3": "ZAF", "uhc_year": 2012, "uhc_type": "insurance"},
    ]
    df = pd.DataFrame(timeline)
    df["treated"] = 1
    fname = RAW_DIR / "uhc_timeline.csv"
    df.to_csv(fname, index=False)
    print(f"\n  UHC Timeline: {len(df)} countries -> {fname}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Data download starting...\n")
    download_worldbank()
    download_who_indicators()
    save_uhc_timeline()
    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
