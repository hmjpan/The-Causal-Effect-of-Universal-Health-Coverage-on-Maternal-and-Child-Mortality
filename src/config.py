"""
Project-wide configuration.
All paths are relative to the project root directory.
Run all scripts from the project root (D:\lunwen\kebiimiansiwang).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR    = PROJECT_ROOT / "data"
RAW_DIR     = DATA_DIR / "raw"
PROC_DIR    = DATA_DIR / "processed"

OUTPUT_DIR  = PROJECT_ROOT / "output"
FIG_DIR     = OUTPUT_DIR / "figures"
TBL_DIR     = OUTPUT_DIR / "tables"

SRC_DIR     = PROJECT_ROOT / "src"

for d in [RAW_DIR, PROC_DIR, FIG_DIR, TBL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Valid treated countries (with >=3 pre-treatment, >=3 post-treatment years in 2000-2021)
EXCLUDE_TREATED = {"LKA", "MYS", "BRA", "COL", "VNM", "MNG", "PHL", "KHM",
                    "KGZ", "THA", "UZB"}

# Micro-states with no NCD mortality data
EXCLUDE_NO_DATA = {"AND", "COK", "DMA", "KNA", "MCO", "MHL", "NIU", "NRU",
                    "PLW", "SMR", "TUV"}

# Study window
YEAR_START = 2000
YEAR_END   = 2021

# Covariates for GSC models
COVARIATES = [
    "log_gdp_per_capita_ppp",
    "urban_population_pct",
    "population_65_plus",
    "health_expenditure_pct_gdp",
]
