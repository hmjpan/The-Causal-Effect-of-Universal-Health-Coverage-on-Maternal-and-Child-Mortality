# UHC and Mortality — Generalized Synthetic Control Analysis

Reproducible analysis code for: *"The Causal Effect of Universal Health Coverage on Maternal and Child Mortality: A Generalized Synthetic Control Analysis of 184 Countries, 2000–2021."*

## Quick Start

```bash
# Clone and run
git clone <this-repo>
cd <repo>
python src/run_all.py
```

That single command runs the entire pipeline: download → process → analyze → visualize.

> **If raw data files already exist**, the download step skips re-downloading. To force fresh data, delete files from `data/raw/` first.  
> **Internet required** for the first run (data is downloaded from WHO and World Bank public APIs).

## Pipeline Steps

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `src/download.py` | Downloads UHC index, mortality rates, DPT3, measles, hospital beds from WHO GHO; 28 covariates from World Bank WDI |
| 2 | `src/fix_codes.py` | Converts World Bank ISO2 codes to ISO3; builds country region classification |
| 3 | `src/build_panel.py` | Merges all data into a balanced country-year panel (184 countries × 22 years) |
| 4 | `src/run_analysis.py` | GSC estimation + event study + factor diagnostics + pre-COVID window + region-matched + in-time placebo |
| 5 | `src/visualize.py` | Generates 4 publication figures and 1 results table |

## Project Structure

```
├── src/
│   ├── config.py              # Shared paths, constants, exclusion lists
│   ├── download.py            # Data download (WHO GHO + World Bank APIs)
│   ├── fix_codes.py           # ISO2→ISO3 + country region classification
│   ├── build_panel.py         # Data processing & panel construction
│   ├── gsc_core.py            # Core GSC algorithm (Bai 2009 IFE estimator)
│   ├── run_analysis.py        # Main analysis + robustness checks
│   ├── visualize.py           # Figures + tables generation
│   └── run_all.py             # Master pipeline runner
├── data/
│   ├── raw/                   # 12 raw data files from WHO & World Bank
│   └── processed/             # analysis_panel.csv + gsc_results.pkl
├── output/
│   ├── figures/               # 4 publication figures (PNG, 300 dpi)
│   └── tables/                # Results table (CSV)
└── README.md
```

## Data Sources

All data are **free, publicly available, no registration required**:

| Source | Indicators | Access |
|--------|-----------|--------|
| WHO Global Health Observatory | UHC Service Coverage Index (SDG 3.8.1), under-5 mortality, maternal mortality, NCD mortality (30-70), life expectancy, DPT3 immunization, measles immunization, hospital beds | [api](https://ghoapi.azureedge.net/api/) |
| World Bank WDI | GDP per capita (PPP), population, urbanization, health expenditure, physicians, nurses, births attended, population 65+, Gini, literacy, PM2.5, fertility, internet, electricity, water, sanitation, food production, agriculture GDP, population density, exports, tax revenue, unemployment | [api](http://api.worldbank.org/v2/) |
| Literature | UHC reform timeline (42 countries, 1950-2020) | derived from peer-reviewed literature |

## Method

**Generalized Synthetic Control (GSC; Xu 2017)** with Bai's (2009) iterative principal components estimator for interactive fixed effects.

The IFE model decomposes panel outcomes into:  
$$Y_{it} = X_{it}'\beta + \lambda_i'F_t + \varepsilon_{it}$$

Estimation proceeds in 3 steps:
1. Estimate IFE on 153 control countries via iterative PC
2. For each of 31 treated countries, estimate unit-specific factor loadings from pre-treatment data
3. Compute ATT as mean(Y_actual − Y_counterfactual) over post-treatment periods

Number of factors ($r$) selected by cross-validation, confirmed by eigenvalue ratio test (Onatski 2010; Ahn-Horenstein 2013).

## Robustness

- Placebo permutation tests (n = 200)
- Leave-one-out donor pool analysis
- Augmented synthetic control (Ben-Michael et al. 2022)
- Region-matched donor pools
- Staggered TWFE DiD
- Pre-treatment window variation (3–15 years)
- Pre-COVID window (2000–2019)
- In-time placebo (5-year backdate)
- Pre-treatment MSPE fit diagnostics
- Rambachan-Roth sensitivity bounds
- Eigenvalue ratio test for factor selection
- Heterogeneity by reform type and baseline UHC

## Requirements

Python 3.9+ with:
```
numpy>=1.24  scipy>=1.11  pandas>=2.0  matplotlib>=3.7  seaborn>=0.12
statsmodels>=0.14  requests>=2.31  pycountry>=22.0
```

Install: `pip install numpy scipy pandas matplotlib seaborn statsmodels requests pycountry`

## Key Results

| Outcome | ATT (GSC, r=1) | Placebo p | Interpretation |
|---------|---------------|-----------|----------------|
| Under-5 mortality | −0·108 | 0·005 | 10·3% reduction |
| Maternal mortality | −0·097 | 0·005 | 9·3% reduction |
| NCD mortality (30–70) | −0·019 | 0·470 | Not significant |

- **Population-weighted ATT**: −0·223 (under-5), −0·225 (maternal) — effects driven by populous nations
- **Equity**: Rich-poor under-5 mortality gap narrowed 49% (47·9 → 24·4 per 1,000)
- **Mechanism**: UHC increases DPT3 immunization by 4·1 percentage points
- **Factor selection**: Eigenvalue ratio = 79·8, supporting r = 1

## License

MIT. Data from WHO and World Bank are subject to their respective terms of use.


