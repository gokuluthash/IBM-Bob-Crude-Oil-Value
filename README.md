# 🛢️ Global Crude Oil Production Intelligence Platform

## Project Overview

A complete, professional-grade Business Intelligence and Data Analytics application built on the OECD/IEA Global Crude Oil Production dataset. The platform transforms raw production data into actionable intelligence for energy analysts, policy makers, and investment strategists.

The application follows the full BI workflow:

```
RAW DATA → CLEAN DATA → INFORMATION → INSIGHT → DECISION → ACTION
```

---

## Business Problem

Global crude oil markets are characterised by high supply concentration, geopolitical dependencies, and long-cycle production dynamics. This application answers five core business intelligence questions:

1. **What is happening?** — Current global and country-level production KPIs
2. **How is it changing?** — Long-term trends, YoY growth, CAGR analysis
3. **Why is it happening?** — Driver analysis across regions and producers
4. **What could go wrong?** — Supply concentration risk, depletion risk, volatility analysis
5. **What should be done?** — Evidence-based recommended actions

---

## Dataset Description

| Attribute         | Value                                   |
|------------------|-----------------------------------------|
| File             | `Crude_oil_Value.csv`                   |
| Source           | OECD/IEA (via OECD.Stat)               |
| Indicator        | OILPROD (Crude Oil Production)          |
| Unit             | KTOE (Kilotonnes of Oil Equivalent)     |
| Frequency        | Annual                                  |
| Total Records    | 8,236                                   |
| Countries/Regions| 142                                     |
| Year Range       | 1960–2017                               |
| Missing Values   | 2,132 (25.9%) — flagged 'L' (estimated) |

### Dataset Source

- OECD iLibrary / OECD.Stat Energy Statistics
- Indicator: `OILPROD` | Subject: `TOT` | Measure: `KTOE` | Frequency: `A`

---

## Dataset Structure

| Column     | Type    | Description                                   |
|------------|---------|-----------------------------------------------|
| LOCATION   | String  | ISO-3 country/region code                    |
| INDICATOR  | String  | Always `OILPROD` (constant)                  |
| SUBJECT    | String  | Always `TOT` — total production (constant)  |
| MEASURE    | String  | Always `KTOE` (constant)                    |
| FREQUENCY  | String  | Always `A` — annual (constant)              |
| TIME       | Integer | Observation year (1960–2017)                 |
| Value      | Float   | Annual oil production in KTOE                |
| Flag Codes | String  | `L` = limited/estimated; blank = actual      |

---

## Technologies Used

| Layer          | Technology          |
|---------------|---------------------|
| Language       | Python 3.11+        |
| Frontend / UI  | Streamlit 1.36.0    |
| Visualisation  | Plotly 5.22.0       |
| Data Processing| Pandas 2.2.2        |
| Numerics       | NumPy 1.26.4        |
| Reporting      | python-docx 1.1.2   |

---

## Project Structure

```
CrudeOilBI/
│
├── app.py                          # Main application (frontend + backend + analysis)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── ProjectReport.docx              # Professional project report
├── generate_report.py              # Script to regenerate the .docx report
└── data/
    └── Crude_oil_Value.csv         # Source dataset
```

---

## Installation & Environment Setup

### 1. Clone / Navigate to the project folder

```bash
cd CrudeOilBI
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser at:

```
http://localhost:8501
```

> **Note:** Ensure the `data/Crude_oil_Value.csv` file is present in the `data/` subdirectory relative to `app.py`.

---

## Dashboard Pages

| Page                          | Content                                                          |
|-------------------------------|------------------------------------------------------------------|
| 📊 Executive Overview          | World KPIs, production trend, decade analysis, top-15 producers |
| 📈 Trends & Country Analysis   | Multi-country line charts, growth leaders & laggards, deep-dive |
| ⚠️ Risk Analysis               | HHI concentration, regional share, volatility, risk register    |
| 🚀 Opportunities & Drivers     | Production lifecycle, emerging producers, opportunity register   |
| 🧠 Executive Insights          | 5 Findings, 3 Risks, 3 Opportunities, 5 Actions                 |
| 🔬 Data Quality & Explorer     | DQ assessment, data dictionary, raw data table with filters     |

---

## Key Findings

1. **World production peaked at ~3,992 MTOE in 2016** — the highest on record in this dataset
2. **Top-5 producers supply >50% of world oil** — structural concentration risk
3. **US Shale Revolution (2008–2017)** transformed the global supply landscape
4. **Legacy producers (UK, Norway, Mexico, Venezuela) are in structural decline**
5. **Africa and CIS are the fastest-growing production regions** since 2000

---

## Principal Risks

1. **Supply Concentration Risk** — top-5 producers >50% share; single-producer disruption can cause global crisis
2. **Depletion of Mature Fields** — UK, Norway, Mexico in multi-decade decline with no recovery path
3. **Data Uncertainty** — 25.9% of records estimated (flag='L'); smaller-country totals uncertain

---

## Opportunities

1. **US Shale Technology Export** — replicable in Argentina, China, Poland, Algeria
2. **African Deep-Water Development** — Angola, Nigeria production ramp possible
3. **Supply Diversification Strategy** — 60+ countries produce; diversification tractable

---

## Recommended Actions

1. Establish a Global Supply Concentration Monitor (HHI threshold alerts)
2. Develop depletion-adjusted net supply forecasts for major producers
3. Prioritise CIS & Africa investment intelligence
4. Build Strategic Petroleum Reserve adequacy models
5. Extend dataset to post-2017 for forward-looking analysis

---

## Limitations

- Dataset ends at 2017; does not cover COVID-19 demand collapse (2020) or 2018–19 shale peak
- No price data — cannot correlate production with oil price cycles (although key price events are annotated)
- No demand data — analysis is supply-only; demand-supply balance cannot be computed
- Sub-annual granularity is not available (annual frequency only)
- Some country data is estimated (flag='L') — quantitative precision limited for smaller producers

---

## Conclusion

This platform provides a complete, evidence-based view of global crude oil production from 1960 to 2017. Every KPI, insight, risk, and opportunity is derived directly from the OECD/IEA dataset with no invented values. The application is suitable for energy analyst portfolios, academic submission, or executive stakeholder demonstrations.

---

*Built with Python · Streamlit · Plotly · Pandas | Data: OECD/IEA OILPROD*
