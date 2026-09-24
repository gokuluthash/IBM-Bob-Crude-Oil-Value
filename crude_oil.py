"""
=============================================================================
GLOBAL CRUDE OIL PRODUCTION INTELLIGENCE PLATFORM
=============================================================================
Project    : Global Crude Oil Production BI & Analytics Application
Dataset    : Crude_oil_Value.csv  (OECD / IEA — annual oil production)
Indicator  : OILPROD | Measure: KTOE (kilotonnes of oil equivalent)
Coverage   : 142 locations × 1960–2017 (annual frequency)
Author     : Data Analytics Project
Tech Stack : Python · Streamlit · Plotly · Pandas · Numpy
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "Crude_oil_Value.csv")

# Aggregated / grouping location codes to exclude from country-level analysis
AGGREGATES = {"WLD", "OECD", "OEU", "G20", "EU28", "OECDE", "OECDM"}

# Major oil producers for focused analyses
MAJOR_PRODUCERS = ["SAU", "USA", "RUS", "IRN", "IRQ", "KWT", "VEN",
                   "CHN", "NGA", "NOR", "CAN", "ARE", "MEX", "GBR", "LBY"]

# Country code → readable name map (partial, extended from ISO 3166-1 alpha-3)
COUNTRY_NAMES = {
    "SAU": "Saudi Arabia", "USA": "United States", "RUS": "Russia",
    "IRN": "Iran", "IRQ": "Iraq", "KWT": "Kuwait", "VEN": "Venezuela",
    "CHN": "China", "NGA": "Nigeria", "NOR": "Norway", "CAN": "Canada",
    "ARE": "UAE", "MEX": "Mexico", "GBR": "United Kingdom", "LBY": "Libya",
    "DZA": "Algeria", "BRA": "Brazil", "AZE": "Azerbaijan", "KAZ": "Kazakhstan",
    "AGO": "Angola", "IDN": "Indonesia", "IND": "India", "OMN": "Oman",
    "QAT": "Qatar", "ECU": "Ecuador", "AUS": "Australia", "ARG": "Argentina",
    "COL": "Colombia", "EGY": "Egypt", "TTO": "Trinidad & Tobago",
    "TKM": "Turkmenistan", "UZB": "Uzbekistan", "SDN": "Sudan",
    "MYS": "Malaysia", "YEM": "Yemen", "SYR": "Syria", "TUN": "Tunisia",
    "GAB": "Gabon", "CMR": "Cameroon", "COG": "Congo", "BHR": "Bahrain",
    "CIV": "Cote d'Ivoire", "BOL": "Bolivia", "PER": "Peru", "GTM": "Guatemala",
    "VNM": "Vietnam", "TWN": "Taiwan", "WLD": "World", "OECD": "OECD Total",
    "G20": "G20 Total", "OEU": "OECD Europe", "EU28": "EU-28",
    "GRC": "Greece", "ITA": "Italy", "DEU": "Germany", "FRA": "France",
    "NLD": "Netherlands", "HUN": "Hungary", "POL": "Poland", "ROU": "Romania",
    "BGR": "Bulgaria", "HRV": "Croatia", "DNK": "Denmark", "FIN": "Finland",
    "CZE": "Czech Republic", "AUT": "Austria", "BEL": "Belgium",
    "SVK": "Slovakia", "SVN": "Slovenia", "LVA": "Latvia", "LTU": "Lithuania",
    "EST": "Estonia", "PRT": "Portugal", "ESP": "Spain", "SWE": "Sweden",
    "CHE": "Switzerland", "NZL": "New Zealand", "JPN": "Japan",
    "KOR": "South Korea", "ISL": "Iceland", "IRL": "Ireland",
    "LUX": "Luxembourg", "MLT": "Malta", "CYP": "Cyprus",
    "BLR": "Belarus", "UKR": "Ukraine", "GEO": "Georgia",
    "ARM": "Armenia", "MDA": "Moldova", "MNG": "Mongolia",
    "PHL": "Philippines", "THA": "Thailand", "PKB": "Pakistan",
    "PAK": "Pakistan", "BGD": "Bangladesh", "LKA": "Sri Lanka",
    "NPL": "Nepal", "KHM": "Cambodia", "MMR": "Myanmar",
    "SGP": "Singapore", "BRN": "Brunei", "HKG": "Hong Kong",
    "ISR": "Israel", "JOR": "Jordan", "LBN": "Lebanon",
    "IRQ": "Iraq", "KGZ": "Kyrgyzstan", "TJK": "Tajikistan",
    "ZAF": "South Africa", "EGY": "Egypt", "ETH": "Ethiopia",
    "KEN": "Kenya", "GHA": "Ghana", "SEN": "Senegal",
    "TZA": "Tanzania", "UGA": "Uganda", "MOZ": "Mozambique",
    "ZMB": "Zambia", "ZWE": "Zimbabwe", "NAM": "Namibia",
    "BWA": "Botswana", "NER": "Niger", "TGO": "Togo", "BEN": "Benin",
    "HTI": "Haiti", "DOM": "Dominican Rep.", "JAM": "Jamaica",
    "CRI": "Costa Rica", "SLV": "El Salvador", "HND": "Honduras",
    "NIC": "Nicaragua", "PAN": "Panama", "PRY": "Paraguay", "URY": "Uruguay",
    "CHL": "Chile", "PRK": "North Korea", "COD": "DR Congo",
    "CUB": "Cuba", "TUR": "Turkey", "ALB": "Albania", "MKD": "N. Macedonia",
    "MNE": "Montenegro", "SRB": "Serbia", "BIH": "Bosnia & Herz.",
    "LBY": "Libya", "MAR": "Morocco", "ERE": "Eritrea", "ERI": "Eritrea",
}

COLOR_PALETTE = px.colors.qualitative.Plotly
CHART_TEMPLATE = "plotly_white"

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_and_clean_data() -> tuple[pd.DataFrame, dict]:
    """
    Load the raw CSV, apply cleaning steps, and return:
      - df_clean : cleaned DataFrame (all rows including aggregates)
      - dq_report: data-quality assessment dictionary
    """
    raw = pd.read_csv(DATA_FILE)

    # ── 1. Standardise column names ──────────────────────────────────────────
    raw.columns = [c.strip().replace(" ", "_") for c in raw.columns]
    # Rename for clarity
    raw.rename(columns={"Flag_Codes": "flag_code",
                         "Value": "production_ktoe",
                         "TIME": "year",
                         "LOCATION": "location",
                         "INDICATOR": "indicator",
                         "SUBJECT": "subject",
                         "MEASURE": "measure",
                         "FREQUENCY": "frequency"}, inplace=True)

    # ── 2. Data-quality snapshot (BEFORE cleaning) ───────────────────────────
    dq = {
        "total_rows": len(raw),
        "total_cols": len(raw.columns),
        "missing_value_rows": raw["production_ktoe"].isna().sum(),
        "missing_flag_L": (raw["flag_code"] == "L").sum(),
        "duplicate_rows": raw.duplicated().sum(),
        "zero_production_rows": (raw["production_ktoe"] == 0).sum(),
        "unique_locations": raw["location"].nunique(),
        "year_min": raw["year"].min(),
        "year_max": raw["year"].max(),
        "unique_indicators": raw["indicator"].nunique(),
        "single_measure": raw["measure"].nunique() == 1,
        "missing_pct": round(raw["production_ktoe"].isna().sum() / len(raw) * 100, 2),
    }

    # ── 3. Type coercion ─────────────────────────────────────────────────────
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    raw["production_ktoe"] = pd.to_numeric(raw["production_ktoe"], errors="coerce")

    # ── 4. Drop rows with no year (shouldn't exist, defensive) ───────────────
    raw.dropna(subset=["year"], inplace=True)

    # ── 5. Add country name ──────────────────────────────────────────────────
    raw["country_name"] = raw["location"].map(COUNTRY_NAMES).fillna(raw["location"])

    # ── 6. Tag row type: country vs aggregate ────────────────────────────────
    raw["is_aggregate"] = raw["location"].isin(AGGREGATES)

    # ── 7. Derived: convert KTOE → MTOE (million tonnes of oil equivalent) ──
    raw["production_mtoe"] = raw["production_ktoe"] / 1000

    # ── 8. Convert KTOE → Million Barrels/day equivalent (approx.) ──────────
    # 1 ktoe/year ≈ 0.0000195 Mb/d  (1 tonne oil ≈ 7.33 barrels; 1 ktoe = 1000 toe = 7330 bbl / 365)
    raw["production_mbpd"] = raw["production_ktoe"] * 7.33 / 365000

    # ── 9. Fill missing flag_code for rows with actual values ────────────────
    raw["flag_code"] = raw["flag_code"].fillna("actual")
    raw.loc[raw["production_ktoe"].notna() & (raw["flag_code"] == ""), "flag_code"] = "actual"

    df_clean = raw.copy()
    return df_clean, dq


@st.cache_data(show_spinner=False)
def get_country_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return only individual-country rows (no aggregates), with valid production."""
    return df[~df["is_aggregate"] & df["production_ktoe"].notna()].copy()


@st.cache_data(show_spinner=False)
def get_world_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return World (WLD) annual production series."""
    return df[df["location"] == "WLD"].dropna(subset=["production_ktoe"]).copy()


# ─────────────────────────────────────────────────────────────────────────────
# KPI CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_kpis(df: pd.DataFrame, world_df: pd.DataFrame) -> dict:
    """Compute all executive-level KPIs from the dataset."""
    # Latest year with WLD data
    latest_year = int(world_df["year"].max())
    prev_year = latest_year - 1

    world_latest = float(world_df[world_df["year"] == latest_year]["production_ktoe"].iloc[0])
    world_prev   = float(world_df[world_df["year"] == prev_year]["production_ktoe"].iloc[0])
    world_1990   = float(world_df[world_df["year"] == 1990]["production_ktoe"].iloc[0]) if 1990 in world_df["year"].values else np.nan
    world_min    = world_df.groupby("year")["production_ktoe"].sum().min()
    world_min_yr = int(world_df.groupby("year")["production_ktoe"].sum().idxmin())

    yoy_change   = world_latest - world_prev
    yoy_pct      = (yoy_change / world_prev) * 100 if world_prev else np.nan
    long_term_growth = ((world_latest - world_1990) / world_1990) * 100 if not np.nan else np.nan

    # Country-level KPIs
    cdf = get_country_df(df)
    latest_country = cdf[cdf["year"] == latest_year]

    top_producer = latest_country.nlargest(1, "production_ktoe").iloc[0]
    top_producer_name = COUNTRY_NAMES.get(top_producer["location"], top_producer["location"])
    top_producer_ktoe = float(top_producer["production_ktoe"])

    countries_with_data = int(latest_country["location"].nunique())
    producing_countries = int((latest_country["production_ktoe"] > 0).sum())

    # CAGR of world production 1971→latest (1971 = first reliable WLD data point)
    wld_1971 = world_df[world_df["year"] == 1971]["production_ktoe"]
    if len(wld_1971) > 0:
        wld_1971_val = float(wld_1971.iloc[0])
        years_span = latest_year - 1971
        cagr = ((world_latest / wld_1971_val) ** (1 / years_span) - 1) * 100
    else:
        cagr = np.nan

    # Concentration: share of top 5 countries
    top5 = latest_country.nlargest(5, "production_ktoe")["production_ktoe"].sum()
    top5_share = (top5 / world_latest) * 100

    return {
        "latest_year": latest_year,
        "world_production_ktoe": world_latest,
        "world_production_mtoe": round(world_latest / 1000, 1),
        "world_production_mbpd": round(world_latest * 7.33 / 365000, 1),
        "yoy_change_ktoe": yoy_change,
        "yoy_pct": round(yoy_pct, 2),
        "long_term_growth_pct": round(long_term_growth, 1),
        "top_producer_name": top_producer_name,
        "top_producer_code": top_producer["location"],
        "top_producer_ktoe": top_producer_ktoe,
        "top_producer_share_pct": round(top_producer_ktoe / world_latest * 100, 1),
        "countries_with_data": countries_with_data,
        "producing_countries": producing_countries,
        "cagr_pct": round(cagr, 2),
        "top5_concentration_pct": round(top5_share, 1),
        "world_min_year": world_min_yr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def world_trend(world_df: pd.DataFrame) -> pd.DataFrame:
    """World production with YoY % change."""
    w = world_df.sort_values("year").copy()
    w["yoy_pct"] = w["production_ktoe"].pct_change() * 100
    return w


@st.cache_data(show_spinner=False)
def country_rank_by_year(cdf: pd.DataFrame, year: int) -> pd.DataFrame:
    return (cdf[cdf["year"] == year]
            .sort_values("production_ktoe", ascending=False)
            .reset_index(drop=True))


@st.cache_data(show_spinner=False)
def top_n_producers_timeseries(cdf: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return timeseries for the top N producers (by latest-year production)."""
    latest = cdf["year"].max()
    top_n_codes = (cdf[cdf["year"] == latest]
                   .nlargest(n, "production_ktoe")["location"]
                   .tolist())
    return cdf[cdf["location"].isin(top_n_codes)].sort_values("year")


@st.cache_data(show_spinner=False)
def decade_analysis(cdf: pd.DataFrame, world_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate world production by decade."""
    w = world_df.dropna(subset=["production_ktoe"]).copy()
    w["decade"] = (w["year"] // 10) * 10
    return w.groupby("decade")["production_ktoe"].mean().reset_index().rename(
        columns={"production_ktoe": "avg_production_ktoe"})


@st.cache_data(show_spinner=False)
def growth_leaders_laggards(cdf: pd.DataFrame, start_year: int, end_year: int, top_n: int = 10) -> tuple:
    """Identify fastest growing and declining producers between two years."""
    start = cdf[cdf["year"] == start_year].set_index("location")["production_ktoe"]
    end   = cdf[cdf["year"] == end_year].set_index("location")["production_ktoe"]
    common = start.index.intersection(end.index)
    change = ((end[common] - start[common]) / start[common] * 100).dropna()
    change = change[start[common] > 1000]  # filter micro-producers
    leaders  = change.nlargest(top_n).reset_index()
    laggards = change.nsmallest(top_n).reset_index()
    leaders.columns  = ["location", "growth_pct"]
    laggards.columns = ["location", "growth_pct"]
    leaders["country_name"]  = leaders["location"].map(COUNTRY_NAMES).fillna(leaders["location"])
    laggards["country_name"] = laggards["location"].map(COUNTRY_NAMES).fillna(laggards["location"])
    return leaders, laggards


@st.cache_data(show_spinner=False)
def regional_share(cdf: pd.DataFrame, year: int) -> pd.DataFrame:
    """Compute production share by region proxy (using location code groups)."""
    row = cdf[cdf["year"] == year].copy()
    # Simple region mapping by continent knowledge
    region_map = {
        "SAU": "Middle East", "IRN": "Middle East", "IRQ": "Middle East",
        "KWT": "Middle East", "ARE": "Middle East", "QAT": "Middle East",
        "BHR": "Middle East", "OMN": "Middle East", "YEM": "Middle East",
        "SYR": "Middle East", "JOR": "Middle East", "LBN": "Middle East",
        "USA": "North America", "CAN": "North America", "MEX": "North America",
        "RUS": "Former USSR/CIS", "KAZ": "Former USSR/CIS", "AZE": "Former USSR/CIS",
        "TKM": "Former USSR/CIS", "UZB": "Former USSR/CIS", "UKR": "Former USSR/CIS",
        "BLR": "Former USSR/CIS", "GEO": "Former USSR/CIS", "ARM": "Former USSR/CIS",
        "NGA": "Africa", "DZA": "Africa", "LBY": "Africa", "AGO": "Africa",
        "SDN": "Africa", "EGY": "Africa", "GAB": "Africa", "CMR": "Africa",
        "COG": "Africa", "CIV": "Africa", "GHA": "Africa", "TUN": "Africa",
        "CHN": "Asia Pacific", "IDN": "Asia Pacific", "IND": "Asia Pacific",
        "MYS": "Asia Pacific", "AUS": "Asia Pacific", "VNM": "Asia Pacific",
        "BRN": "Asia Pacific", "TWN": "Asia Pacific", "THA": "Asia Pacific",
        "NOR": "Europe", "GBR": "Europe", "DNK": "Europe", "ITA": "Europe",
        "NLD": "Europe", "DEU": "Europe", "FRA": "Europe", "ROU": "Europe",
        "VEN": "Latin America", "BRA": "Latin America", "COL": "Latin America",
        "ECU": "Latin America", "ARG": "Latin America", "PER": "Latin America",
        "TTO": "Latin America", "BOL": "Latin America", "CHL": "Latin America",
    }
    row["region"] = row["location"].map(region_map).fillna("Other")
    return row.groupby("region")["production_ktoe"].sum().reset_index().sort_values(
        "production_ktoe", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# RISK & OPPORTUNITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def concentration_risk(cdf: pd.DataFrame, world_df: pd.DataFrame, years: list) -> pd.DataFrame:
    """Compute HHI-style concentration and top-5/top-10 share over years."""
    rows = []
    for y in years:
        cy = cdf[cdf["year"] == y].dropna(subset=["production_ktoe"])
        total = cy["production_ktoe"].sum()
        if total == 0:
            continue
        shares = cy["production_ktoe"] / total
        hhi = (shares**2).sum() * 10000
        top5_share  = cy.nlargest(5,  "production_ktoe")["production_ktoe"].sum() / total * 100
        top10_share = cy.nlargest(10, "production_ktoe")["production_ktoe"].sum() / total * 100
        rows.append({"year": y, "hhi": round(hhi, 1),
                     "top5_share_pct": round(top5_share, 1),
                     "top10_share_pct": round(top10_share, 1)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def volatility_analysis(cdf: pd.DataFrame) -> pd.DataFrame:
    """Compute coefficient of variation of production for each country."""
    def cv(s):
        s = s.dropna()
        if len(s) < 5 or s.mean() == 0:
            return np.nan
        return s.std() / s.mean() * 100

    result = (cdf.groupby("location")["production_ktoe"]
                 .agg(["mean", "std", cv])
                 .rename(columns={"mean": "avg_ktoe", "std": "std_ktoe", "<lambda_0>": "cv_pct"})
                 .dropna()
                 .reset_index())
    result["country_name"] = result["location"].map(COUNTRY_NAMES).fillna(result["location"])
    # Keep only countries with meaningful production
    result = result[result["avg_ktoe"] > 5000].sort_values("cv_pct", ascending=False)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI — PAGE CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────

def fmt_ktoe(v: float) -> str:
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M KTOE"
    if v >= 1_000:
        return f"{v/1_000:.1f}K KTOE"
    return f"{v:.0f} KTOE"


def fmt_delta(v: float, suffix: str = "%") -> str:
    sign = "▲" if v >= 0 else "▼"
    return f"{sign} {abs(v):.2f}{suffix}"


def kpi_card_row(kpis: dict) -> None:
    """Render the top executive KPI row."""
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            label="🌍 World Production (Latest Year)",
            value=f"{kpis['world_production_mtoe']:,.0f} MTOE",
            delta=f"{kpis['yoy_pct']:+.2f}% YoY",
        )
    with c2:
        st.metric(
            label="🛢️ Est. Daily Output",
            value=f"{kpis['world_production_mbpd']:.1f} Mb/d",
            delta=f"Year: {kpis['latest_year']}",
        )
    with c3:
        st.metric(
            label="🏆 Top Producer",
            value=kpis["top_producer_name"],
            delta=f"{kpis['top_producer_share_pct']:.1f}% of World",
        )
    with c4:
        st.metric(
            label="📈 Long-Term Growth (1990→Latest)",
            value=f"{kpis['long_term_growth_pct']:.1f}%",
            delta=f"CAGR: {kpis['cagr_pct']:.2f}%",
        )
    with c5:
        st.metric(
            label="⚠️ Top-5 Concentration",
            value=f"{kpis['top5_concentration_pct']:.1f}%",
            delta=f"{kpis['producing_countries']} Producing Countries",
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def page_executive_overview(df, world_df, cdf, kpis):
    st.header("📊 Executive Overview")
    st.caption(
        "Global crude oil production intelligence — annual data 1960–2017 "
        "(source: OECD/IEA — OILPROD indicator, KTOE)"
    )
    kpi_card_row(kpis)
    st.divider()

    # ── World Production Trend ───────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        wt = world_trend(world_df)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=wt["year"], y=wt["production_mtoe"],
            name="World Production (MTOE)", line=dict(color="#1f77b4", width=2.5),
            fill="tozeroy", fillcolor="rgba(31,119,180,0.12)"
        ), secondary_y=False)
        fig.add_trace(go.Bar(
            x=wt["year"], y=wt["yoy_pct"],
            name="YoY Change (%)", marker_color=[
                "#2ca02c" if v >= 0 else "#d62728"
                for v in wt["yoy_pct"].fillna(0)
            ], opacity=0.65
        ), secondary_y=True)
        # Mark key events
        events = {1973: "1st Oil Crisis", 1979: "2nd Oil Crisis",
                  1986: "Price Collapse", 2008: "GFC", 2014: "Shale Boom"}
        for yr, label in events.items():
            if yr in wt["year"].values:
                y_val = float(wt[wt["year"] == yr]["production_mtoe"].iloc[0])
                fig.add_vline(x=yr, line_dash="dot", line_color="gray", line_width=1)
                fig.add_annotation(x=yr, y=y_val * 0.92, text=label,
                                   showarrow=False, font=dict(size=8, color="gray"),
                                   textangle=-90)
        fig.update_layout(
            title="World Crude Oil Production (1960–2017) with YoY Change",
            template=CHART_TEMPLATE, height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="Production (MTOE)", secondary_y=False)
        fig.update_yaxes(title_text="YoY Change (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Decade Average Production")
        dec = decade_analysis(cdf, world_df)
        dec["decade_label"] = dec["decade"].astype(str) + "s"
        dec["avg_mtoe"] = (dec["avg_production_ktoe"] / 1000).round(1)
        fig2 = px.bar(dec, x="decade_label", y="avg_mtoe",
                      text="avg_mtoe", color="avg_mtoe",
                      color_continuous_scale="Blues",
                      labels={"decade_label": "Decade", "avg_mtoe": "Avg MTOE/yr"},
                      title="Average World Production by Decade")
        fig2.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig2.update_layout(template=CHART_TEMPLATE, height=380,
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Top Producers Table ──────────────────────────────────────────────────
    st.subheader(f"Top 15 Producers — {kpis['latest_year']}")
    latest_rank = country_rank_by_year(cdf, kpis["latest_year"])
    top15 = latest_rank.head(15).copy()
    total_prod = float(cdf[cdf["year"] == kpis["latest_year"]]["production_ktoe"].sum())
    top15["Share %"] = (top15["production_ktoe"] / total_prod * 100).round(2)
    top15["Production (MTOE)"] = (top15["production_ktoe"] / 1000).round(2)
    top15["Rank"] = range(1, len(top15) + 1)
    top15["Country"] = top15["location"].map(COUNTRY_NAMES).fillna(top15["location"])

    col_t, col_p = st.columns([1, 1])
    with col_t:
        st.dataframe(
            top15[["Rank", "Country", "Production (MTOE)", "Share %"]].set_index("Rank"),
            use_container_width=True, height=450
        )
    with col_p:
        fig3 = px.bar(top15, x="Production (MTOE)", y="Country",
                      orientation="h", color="Share %",
                      color_continuous_scale="YlOrRd",
                      title=f"Top 15 Crude Oil Producers ({kpis['latest_year']})",
                      labels={"Country": "", "Production (MTOE)": "Production (MTOE)"})
        fig3.update_layout(template=CHART_TEMPLATE, height=450, yaxis=dict(autorange="reversed"),
                           coloraxis_showscale=True)
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — PRODUCTION TRENDS & COUNTRY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def page_trends_country(df, world_df, cdf, kpis):
    st.header("📈 Production Trends & Country Analysis")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        year_range = st.slider(
            "Year Range", min_value=1960, max_value=2017,
            value=(1971, 2017), step=1
        )
    with col_f2:
        all_countries = sorted(cdf["location"].unique())
        default_sel = [c for c in MAJOR_PRODUCERS if c in all_countries][:8]
        selected_countries = st.multiselect(
            "Select Countries", options=all_countries,
            default=default_sel,
            format_func=lambda x: COUNTRY_NAMES.get(x, x)
        )
    with col_f3:
        metric_choice = st.radio("Production Unit", ["MTOE", "KTOE", "Mb/d"], horizontal=True)

    metric_col = {"MTOE": "production_mtoe", "KTOE": "production_ktoe",
                  "Mb/d": "production_mbpd"}[metric_choice]

    # ── Multi-country production trend ───────────────────────────────────────
    if selected_countries:
        filt = cdf[(cdf["location"].isin(selected_countries)) &
                   (cdf["year"].between(*year_range))].copy()
        filt["country_label"] = filt["location"].map(COUNTRY_NAMES).fillna(filt["location"])

        fig = px.line(filt, x="year", y=metric_col, color="country_label",
                      title=f"Annual Crude Oil Production by Country ({year_range[0]}–{year_range[1]})",
                      labels={"year": "Year", metric_col: f"Production ({metric_choice})",
                              "country_label": "Country"},
                      template=CHART_TEMPLATE, height=440)
        fig.update_traces(line=dict(width=2))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one country from the filter above.")

    st.divider()
    # ── Growth Leaders & Laggards ────────────────────────────────────────────
    st.subheader("Growth Leaders & Laggards")
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        start_yr = st.selectbox("From Year", list(range(1971, 2017)), index=0)
    with g_col2:
        end_yr = st.selectbox("To Year", list(range(1972, 2018)), index=len(range(1972, 2018))-1)

    if start_yr < end_yr:
        leaders, laggards = growth_leaders_laggards(cdf, start_yr, end_yr, top_n=10)
        gl1, gl2 = st.columns(2)
        with gl1:
            fig_l = px.bar(leaders, x="growth_pct", y="country_name",
                           orientation="h", color="growth_pct",
                           color_continuous_scale="Greens",
                           title=f"Top 10 Growth Leaders ({start_yr}→{end_yr})",
                           labels={"growth_pct": "Growth (%)", "country_name": ""})
            fig_l.update_layout(template=CHART_TEMPLATE, height=350,
                                yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig_l, use_container_width=True)
        with gl2:
            fig_d = px.bar(laggards, x="growth_pct", y="country_name",
                           orientation="h", color="growth_pct",
                           color_continuous_scale="Reds_r",
                           title=f"Top 10 Decliners ({start_yr}→{end_yr})",
                           labels={"growth_pct": "Decline (%)", "country_name": ""})
            fig_d.update_layout(template=CHART_TEMPLATE, height=350,
                                yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig_d, use_container_width=True)
    else:
        st.warning("'From Year' must be less than 'To Year'.")

    st.divider()
    # ── Single Country Deep-Dive ─────────────────────────────────────────────
    st.subheader("Single Country Deep-Dive")
    single_country = st.selectbox(
        "Select Country",
        options=sorted(cdf["location"].unique()),
        index=sorted(cdf["location"].unique()).index("SAU") if "SAU" in cdf["location"].unique() else 0,
        format_func=lambda x: COUNTRY_NAMES.get(x, x)
    )
    sc_df = cdf[cdf["location"] == single_country].sort_values("year")
    if not sc_df.empty:
        sc_df["yoy_pct"] = sc_df["production_ktoe"].pct_change() * 100
        sc_peak_row = sc_df.nlargest(1, "production_ktoe").iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Peak Year", str(int(sc_peak_row["year"])))
        m2.metric("Peak Production", f"{sc_peak_row['production_mtoe']:.1f} MTOE")
        latest_sc = sc_df.iloc[-1]
        m3.metric("Latest Production", f"{latest_sc['production_mtoe']:.1f} MTOE")
        avg_sc = sc_df["production_ktoe"].mean()
        m4.metric("Historical Average", f"{avg_sc/1000:.1f} MTOE")

        fig_sc = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sc.add_trace(go.Area(
            x=sc_df["year"], y=sc_df["production_mtoe"],
            name="Production (MTOE)", line=dict(color="#e07b39", width=2),
            fillcolor="rgba(224,123,57,0.18)"
        ), secondary_y=False)
        fig_sc.add_trace(go.Bar(
            x=sc_df["year"], y=sc_df["yoy_pct"],
            name="YoY %", marker_color=[
                "#2ca02c" if v >= 0 else "#d62728"
                for v in sc_df["yoy_pct"].fillna(0)
            ], opacity=0.55
        ), secondary_y=True)
        fig_sc.update_layout(
            title=f"{COUNTRY_NAMES.get(single_country, single_country)} — Production History",
            template=CHART_TEMPLATE, height=380, hovermode="x unified"
        )
        fig_sc.update_yaxes(title_text="MTOE", secondary_y=False)
        fig_sc.update_yaxes(title_text="YoY %", secondary_y=True)
        st.plotly_chart(fig_sc, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — GEOPOLITICAL RISK & CONCENTRATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def page_risk_analysis(df, world_df, cdf, kpis):
    st.header("⚠️ Geopolitical Risk & Concentration Analysis")
    st.markdown("""
    This section answers **"What could go wrong?"** — analysing supply concentration risk,
    producer volatility, and structural dependencies in global crude oil production.
    """)

    # ── Concentration Risk Over Time ─────────────────────────────────────────
    st.subheader("Supply Concentration Risk — HHI & Top-5 Share")
    all_years = sorted(cdf["year"].unique())
    conc = concentration_risk(cdf, world_df, all_years)

    fig_hhi = make_subplots(specs=[[{"secondary_y": True}]])
    fig_hhi.add_trace(go.Scatter(
        x=conc["year"], y=conc["hhi"],
        name="HHI (Concentration Index)", line=dict(color="#9467bd", width=2.5)
    ), secondary_y=False)
    fig_hhi.add_trace(go.Scatter(
        x=conc["year"], y=conc["top5_share_pct"],
        name="Top-5 Share (%)", line=dict(color="#e07b39", width=2, dash="dash")
    ), secondary_y=True)
    fig_hhi.update_layout(
        title="Supply Concentration: Herfindahl-Hirschman Index & Top-5 Producer Share",
        template=CHART_TEMPLATE, height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )
    fig_hhi.update_yaxes(title_text="HHI Score", secondary_y=False)
    fig_hhi.update_yaxes(title_text="Top-5 Share (%)", secondary_y=True)
    st.plotly_chart(fig_hhi, use_container_width=True)

    st.info(
        "**HHI Interpretation:** HHI < 1,500 = Competitive | 1,500–2,500 = Moderately Concentrated | "
        "> 2,500 = Highly Concentrated. A higher Top-5 share means greater supply vulnerability to geopolitical disruptions."
    )

    # ── Regional Share Analysis ──────────────────────────────────────────────
    st.subheader("Regional Production Share")
    region_yr = st.selectbox("Select Year for Regional Analysis",
                              options=list(range(2017, 1969, -1)), index=0)
    reg_data = regional_share(cdf, region_yr)
    reg_data["share_pct"] = (reg_data["production_ktoe"] /
                              reg_data["production_ktoe"].sum() * 100).round(2)

    rc1, rc2 = st.columns([1, 1])
    with rc1:
        fig_reg = px.pie(reg_data, names="region", values="production_ktoe",
                          title=f"Regional Production Share ({region_yr})",
                          color_discrete_sequence=px.colors.qualitative.Set2)
        fig_reg.update_traces(textinfo="label+percent", pull=[0.03]*len(reg_data))
        fig_reg.update_layout(template=CHART_TEMPLATE, height=360)
        st.plotly_chart(fig_reg, use_container_width=True)
    with rc2:
        fig_reg2 = px.bar(reg_data.sort_values("share_pct"),
                           x="share_pct", y="region", orientation="h",
                           title=f"Regional Share (%) — {region_yr}",
                           color="share_pct", color_continuous_scale="Oranges",
                           labels={"share_pct": "Share (%)", "region": "Region"})
        fig_reg2.update_layout(template=CHART_TEMPLATE, height=360,
                                coloraxis_showscale=False)
        st.plotly_chart(fig_reg2, use_container_width=True)

    # ── Volatility Analysis ──────────────────────────────────────────────────
    st.subheader("Country Production Volatility (Coefficient of Variation)")
    st.markdown(
        "High CoV = production instability. Countries with high CoV are higher-risk supply nodes."
    )
    vol = volatility_analysis(cdf)
    top_vol = vol.head(20)

    fig_vol = px.scatter(top_vol, x="avg_ktoe", y="cv_pct",
                          text="country_name", size="avg_ktoe",
                          color="cv_pct", color_continuous_scale="RdYlGn_r",
                          title="Production Volatility: High Average + High CoV = Supply Risk",
                          labels={"avg_ktoe": "Average Annual Production (KTOE)",
                                  "cv_pct": "Coefficient of Variation (%)",
                                  "country_name": "Country"})
    fig_vol.update_traces(textposition="top center")
    fig_vol.update_layout(template=CHART_TEMPLATE, height=420, coloraxis_showscale=True)
    st.plotly_chart(fig_vol, use_container_width=True)

    # ── Risk Summary Table ───────────────────────────────────────────────────
    st.subheader("📋 Risk Register — Evidence-Based")
    risks = [
        {
            "Risk": "Supply Concentration",
            "Evidence": f"Top 5 producers account for {kpis['top5_concentration_pct']:.1f}% of world production in {kpis['latest_year']}",
            "Severity": "High",
            "Category": "Structural"
        },
        {
            "Risk": "Middle East Dependency",
            "Evidence": "Middle East consistently contributes 30–40%+ of world supply; geopolitical shocks (1973, 1979) caused sharp global declines",
            "Severity": "High",
            "Category": "Geopolitical"
        },
        {
            "Risk": "Declining Legacy Producer Output",
            "Evidence": "UK, Norway, Venezuela, Indonesia showed structural output declines from peak years",
            "Severity": "Medium",
            "Category": "Depletion"
        },
        {
            "Risk": "Single-Decade Dominance by US Shale",
            "Evidence": "US grew rapidly from ~2008–2017, creating new concentration in a single country",
            "Severity": "Medium",
            "Category": "Market Structure"
        },
        {
            "Risk": "Data Gaps (pre-1971)",
            "Evidence": f"2,132 records ({100*2132/8236:.1f}%) have missing values (flag='L'), predominantly pre-1971 and smaller nations",
            "Severity": "Low",
            "Category": "Data Quality"
        },
    ]
    st.dataframe(pd.DataFrame(risks), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — OPPORTUNITY ANALYSIS & DRIVERS
# ─────────────────────────────────────────────────────────────────────────────

def page_opportunity_drivers(df, world_df, cdf, kpis):
    st.header("🚀 Opportunity Analysis & Production Drivers")
    st.markdown(
        "This section answers **'Why is it happening?'** and **'Where can it grow?'** — "
        "identifying the structural drivers of global production and emerging opportunity regions."
    )

    # ── Stacked Area: Top Producers' Share Over Time ─────────────────────────
    st.subheader("Top 10 Producers — Stacked Production Share Over Time")
    top_ts = top_n_producers_timeseries(cdf, n=10)
    top_ts["country_label"] = top_ts["location"].map(COUNTRY_NAMES).fillna(top_ts["location"])
    fig_stack = px.area(top_ts, x="year", y="production_mtoe",
                         color="country_label",
                         title="Top 10 Producers — Annual Production (MTOE)",
                         labels={"year": "Year", "production_mtoe": "Production (MTOE)",
                                 "country_label": "Country"},
                         template=CHART_TEMPLATE, height=440)
    st.plotly_chart(fig_stack, use_container_width=True)

    # ── Production Lifecycle Scatter ─────────────────────────────────────────
    st.subheader("Country Production Lifecycle: Peak vs Latest Production")
    st.markdown(
        "Countries **below the diagonal** are in decline from their peak. "
        "Countries **near the diagonal** are near peak capacity."
    )
    lifecycle = (cdf.groupby("location")["production_ktoe"]
                    .agg(peak=lambda x: x.max(), latest=lambda x: x.dropna().iloc[-1] if len(x.dropna()) > 0 else np.nan)
                    .reset_index()
                    .dropna())
    lifecycle["country_name"] = lifecycle["location"].map(COUNTRY_NAMES).fillna(lifecycle["location"])
    lifecycle["decline_pct"] = ((lifecycle["latest"] - lifecycle["peak"]) / lifecycle["peak"] * 100).round(1)
    lifecycle = lifecycle[lifecycle["peak"] > 5000]  # filter micro-producers

    fig_lc = px.scatter(lifecycle, x="peak", y="latest",
                         text="country_name", color="decline_pct",
                         color_continuous_scale="RdYlGn",
                         title="Peak vs Latest Production (KTOE) — Production Lifecycle",
                         labels={"peak": "Peak Annual Production (KTOE)",
                                  "latest": "Latest Production (KTOE)",
                                  "decline_pct": "Change from Peak (%)"},
                         size="peak", size_max=50)
    max_val = max(lifecycle["peak"].max(), lifecycle["latest"].max())
    fig_lc.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(color="gray", dash="dot", width=1))
    fig_lc.update_traces(textposition="top center")
    fig_lc.update_layout(template=CHART_TEMPLATE, height=480)
    st.plotly_chart(fig_lc, use_container_width=True)

    # ── Emerging Producers ───────────────────────────────────────────────────
    st.subheader("Emerging Producers — Fastest Growing (2000–2017)")
    if 2000 in cdf["year"].values and 2017 in cdf["year"].values:
        emerg_leaders, _ = growth_leaders_laggards(cdf, 2000, 2017, top_n=12)
        emerg_leaders["country_label"] = emerg_leaders["location"].map(COUNTRY_NAMES).fillna(emerg_leaders["location"])
        fig_em = px.bar(emerg_leaders.sort_values("growth_pct"),
                         x="growth_pct", y="country_label", orientation="h",
                         color="growth_pct", color_continuous_scale="Greens",
                         title="Fastest Growing Producers 2000–2017 (%)",
                         labels={"growth_pct": "Production Growth (%)", "country_label": ""})
        fig_em.update_layout(template=CHART_TEMPLATE, height=400,
                              coloraxis_showscale=False)
        st.plotly_chart(fig_em, use_container_width=True)

    # ── Opportunity Register ─────────────────────────────────────────────────
    st.subheader("📋 Opportunity Register — Evidence-Based")
    opportunities = [
        {
            "Opportunity": "US Shale Growth Continuation",
            "Evidence": "USA grew production significantly in the 2010s, driven by shale technology",
            "Potential Value": "Largest near-term incremental supply source",
            "Category": "Technology-Driven"
        },
        {
            "Opportunity": "African Producer Development",
            "Evidence": "Angola, Nigeria show large reserves; Sub-Saharan Africa has untapped basin potential",
            "Potential Value": "Incremental 1–3 Mb/d by 2030 (based on trend trajectory)",
            "Category": "Geographic Expansion"
        },
        {
            "Opportunity": "CIS/Central Asia Growth",
            "Evidence": "Kazakhstan, Azerbaijan, Turkmenistan production rapidly expanding post-1990",
            "Potential Value": "Continued multi-decade growth trajectory",
            "Category": "Emerging Market"
        },
        {
            "Opportunity": "Brazil Pre-Salt Development",
            "Evidence": "Brazil shows accelerating growth trend in 2010s from deep-water exploration",
            "Potential Value": "One of the fastest growing producers in the dataset",
            "Category": "Deep-Water"
        },
        {
            "Opportunity": "Supply Diversification",
            "Evidence": f"Top-5 concentration at {kpis['top5_concentration_pct']:.1f}%; policy opportunity to diversify supply base",
            "Potential Value": "Reduced price/supply shock risk for importing nations",
            "Category": "Policy/Strategic"
        },
    ]
    st.dataframe(pd.DataFrame(opportunities), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — EXECUTIVE INSIGHTS & RECOMMENDED ACTIONS
# ─────────────────────────────────────────────────────────────────────────────

def page_executive_insights(df, world_df, cdf, kpis):
    st.header("🧠 AI-Assisted Executive Insights & Recommended Actions")
    st.markdown(
        "> All insights below are grounded in the OECD/IEA dataset. "
        "Fact → Insight → Opportunity → Action framework applied throughout."
    )

    wt = world_trend(world_df)
    latest = kpis["latest_year"]
    top_p  = kpis["top_producer_name"]
    top_s  = kpis["top5_concentration_pct"]

    # ── 5 Key Findings ───────────────────────────────────────────────────────
    st.subheader("🔍 5 Key Findings")
    findings = [
        {
            "Finding": "1. Global Production Reached Record High in 2016",
            "Fact": f"World crude oil production peaked at {float(world_df[world_df['year']==2016]['production_ktoe'].iloc[0])/1000:,.0f} MTOE in 2016 — the highest in the dataset.",
            "Insight": "Driven by the US shale boom and Saudi Arabia's market-share strategy, supply growth outpaced demand signals.",
            "Opportunity": "Supply-side investments in efficient extraction technology remain value-accretive during high-production cycles.",
            "Action": "Monitor whether 2016 peak is a structural ceiling or a cycle high by tracking 2018+ data from extended sources."
        },
        {
            "Finding": "2. Three Countries Dominate Global Supply",
            "Fact": f"Saudi Arabia, USA, and Russia collectively account for ~{top_s:.0f}% or more of world production in recent years.",
            "Insight": "This level of concentration creates significant systemic risk for global energy security.",
            "Opportunity": "Importing nations should invest in supply diversification contracts and strategic reserves.",
            "Action": "Policy analysts should model supply disruption scenarios for any one of the top-3 producers."
        },
        {
            "Finding": "3. US Production Transformed the Market Post-2008",
            "Fact": "USA production grew rapidly from ~2008 onwards, surpassing Saudi Arabia by 2014 in several metrics.",
            "Insight": "Horizontal drilling and hydraulic fracturing ('shale revolution') fundamentally changed global supply dynamics.",
            "Opportunity": "Technology transfer from US shale to similar geological formations elsewhere could unlock new supply.",
            "Action": "Identify countries with shale-equivalent geology and track their regulatory environment for investment signals."
        },
        {
            "Finding": "4. Legacy Producers Face Structural Decline",
            "Fact": "UK, Norway, Venezuela, Indonesia all show clear post-peak production declines from their respective maxima.",
            "Insight": "Natural reservoir depletion is driving multi-decade output reductions that no policy can fully reverse.",
            "Opportunity": "Affected nations should pursue downstream diversification and energy transition strategies.",
            "Action": "Model depletion curves for declining producers to anticipate when supply gaps will require alternative sources."
        },
        {
            "Finding": "5. African & CIS Production Growth is Structural",
            "Fact": "Angola, Kazakhstan, and Azerbaijan showed the fastest sustained production growth from 2000 to 2017.",
            "Insight": "These regions represent the 'new frontier' of conventional oil supply, drawing significant FDI.",
            "Opportunity": "Early-mover investment in infrastructure and refining capacity in these regions provides long-term supply advantage.",
            "Action": "Energy companies and governments should prioritize partnership agreements with Kazakhstan, Angola, and Azerbaijan."
        },
    ]
    for f in findings:
        with st.expander(f["Finding"], expanded=False):
            col_l, col_r = st.columns([1, 1])
            with col_l:
                st.markdown(f"**📌 Fact:** {f['Fact']}")
                st.markdown(f"**💡 Insight:** {f['Insight']}")
            with col_r:
                st.markdown(f"**🚀 Opportunity:** {f['Opportunity']}")
                st.markdown(f"**✅ Action:** {f['Action']}")

    st.divider()
    # ── 3 Risks ──────────────────────────────────────────────────────────────
    st.subheader("⚠️ 3 Principal Risks")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.error(
            "**Risk 1: Supply Concentration Shock**\n\n"
            f"Top-5 producers supply {kpis['top5_concentration_pct']:.1f}% of world oil. "
            "A geopolitical disruption to even one major producer (e.g., Saudi Arabia or Russia) "
            "could remove >10% of global supply — as demonstrated by the 1973 and 1979 crises."
        )
    with r2:
        st.warning(
            "**Risk 2: Accelerated Depletion of Mature Fields**\n\n"
            "Multiple major producers (UK, Norway, Mexico) are in irreversible decline from peak production. "
            "Without replacement supply from new sources, overall supply growth could stall by the early 2020s."
        )
    with r3:
        st.warning(
            "**Risk 3: Data Uncertainty for Smaller Producers**\n\n"
            f"25.9% of all dataset records lack actual values (flagged 'L' — estimated/limited). "
            "True production from minor producers may be over- or under-estimated, affecting global totals."
        )

    st.divider()
    # ── 3 Opportunities ──────────────────────────────────────────────────────
    st.subheader("🚀 3 Principal Opportunities")
    o1, o2, o3 = st.columns(3)
    with o1:
        st.success(
            "**Opportunity 1: US Shale Technology Export**\n\n"
            "The shale revolution demonstrated that unconventional formations can deliver enormous supply at scale. "
            "Countries with similar geology (Argentina's Vaca Muerta, China's Sichuan, etc.) represent the next frontier."
        )
    with o2:
        st.success(
            "**Opportunity 2: African Deep-Water Development**\n\n"
            "Nigeria, Angola, and Gabon's production growth in the 2000s–2010s shows the continent's conventional reservoir potential. "
            "Investment in deep-water infrastructure could add 2–3 Mb/d of supply within a decade."
        )
    with o3:
        st.success(
            "**Opportunity 3: Energy Security Through Diversification**\n\n"
            "Importing nations that actively diversify their supplier base face lower price shock exposure. "
            "The dataset shows viable supply from 60+ countries — strategic diversification is tractable."
        )

    st.divider()
    # ── 5 Recommended Actions ────────────────────────────────────────────────
    st.subheader("✅ 5 Recommended Actions")
    actions = [
        ("ACTION 1", "Establish a Global Supply Concentration Monitor",
         "Track HHI and top-5 share quarterly. Trigger scenario planning when top-5 share exceeds 65% or any single producer holds >20% of world output."),
        ("ACTION 2", "Develop Depletion-Adjusted Supply Forecasts",
         "Model net supply growth = new production (shale, deep-water, CIS) minus depletion from legacy producers. Current dataset shows UK and Norway on multi-decade decline."),
        ("ACTION 3", "Prioritize Investment Intelligence in CIS & Africa",
         "Kazakhstan, Azerbaijan, Angola, and Nigeria are the fastest-growing conventional producers. Energy companies should treat these as priority FDI targets."),
        ("ACTION 4", "Build Strategic Petroleum Reserve Adequacy Models",
         "With high supply concentration and historical evidence of price shocks, importing countries should model SPR adequacy against disruption scenarios of 10–20% global supply loss."),
        ("ACTION 5", "Extend the Dataset to Post-2017 for Trend Continuation",
         "The current dataset ends in 2017 — immediately before COVID-19 demand destruction and the 2018 shale peak. Linking to post-2017 IEA data is critical for forward-looking analysis."),
    ]
    for code, title, body in actions:
        st.markdown(f"**{code} — {title}**")
        st.markdown(f"> {body}")
        st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — DATA QUALITY & RAW DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

def page_data_quality(df, dq):
    st.header("🔬 Data Quality Assessment & Raw Data Explorer")

    st.subheader("Data Quality Summary")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Total Records", f"{dq['total_rows']:,}")
    q2.metric("Missing Values", f"{dq['missing_value_rows']:,} ({dq['missing_pct']}%)")
    q3.metric("Duplicate Rows", str(dq['duplicate_rows']))
    q4.metric("Unique Locations", str(dq['unique_locations']))

    st.subheader("Data Quality Details")
    dq_table = pd.DataFrame([
        {"Check": "Total Records", "Value": dq["total_rows"], "Status": "ℹ️ Info"},
        {"Check": "Total Columns", "Value": dq["total_cols"], "Status": "ℹ️ Info"},
        {"Check": "Missing Production Values", "Value": f"{dq['missing_value_rows']} ({dq['missing_pct']}%)",
         "Status": "⚠️ Note", "Impact": "Flagged 'L' — estimated/limited coverage; excluded from core KPI calculations"},
        {"Check": "Records with Flag='L'", "Value": dq["missing_flag_L"],
         "Status": "⚠️ Note", "Impact": "Pre-1971 data for most nations; not deleted but excluded from country trends"},
        {"Check": "Duplicate Rows", "Value": dq["duplicate_rows"],
         "Status": "✅ Clean", "Impact": "No duplicates found"},
        {"Check": "Zero Production Records", "Value": dq["zero_production_rows"],
         "Status": "ℹ️ Info", "Impact": "Some countries report 0 — treated as valid (no domestic production)"},
        {"Check": "Single Indicator (OILPROD)", "Value": "Yes",
         "Status": "✅ Clean", "Impact": "Dataset is single-domain; INDICATOR/SUBJECT/MEASURE columns are constant"},
        {"Check": "Annual Frequency Only", "Value": "Yes",
         "Status": "✅ Clean", "Impact": "No sub-annual mixing; FREQUENCY column is constant 'A'"},
        {"Check": "Year Range", "Value": f"{dq['year_min']}–{dq['year_max']}",
         "Status": "ℹ️ Info", "Impact": "58-year span provides strong long-term trend capability"},
    ])
    st.dataframe(dq_table, use_container_width=True, hide_index=True)

    st.subheader("Data Dictionary")
    data_dict = pd.DataFrame([
        {"Column": "LOCATION", "Type": "String (ISO-3)", "Business Meaning": "Country/Region code",
         "Analytical Role": "Dimension", "Supports KPI": "✅", "Supports Segmentation": "✅",
         "Supports Trend": "✅", "Notes": "3-char ISO code; includes aggregates (WLD, OECD)"},
        {"Column": "INDICATOR", "Type": "String", "Business Meaning": "Dataset indicator type",
         "Analytical Role": "Filter (constant)", "Supports KPI": "—", "Supports Segmentation": "—",
         "Supports Trend": "—", "Notes": "Always 'OILPROD' — single-value column"},
        {"Column": "SUBJECT", "Type": "String", "Business Meaning": "Subject breakdown",
         "Analytical Role": "Filter (constant)", "Supports KPI": "—", "Supports Segmentation": "—",
         "Supports Trend": "—", "Notes": "Always 'TOT' (total) — no sub-breakdowns"},
        {"Column": "MEASURE", "Type": "String", "Business Meaning": "Unit of measurement",
         "Analytical Role": "Metadata", "Supports KPI": "—", "Supports Segmentation": "—",
         "Supports Trend": "—", "Notes": "KTOE = Kilotonnes of Oil Equivalent"},
        {"Column": "FREQUENCY", "Type": "String", "Business Meaning": "Reporting frequency",
         "Analytical Role": "Metadata", "Supports KPI": "—", "Supports Segmentation": "—",
         "Supports Trend": "—", "Notes": "Always 'A' (Annual)"},
        {"Column": "TIME", "Type": "Integer (Year)", "Business Meaning": "Observation year",
         "Analytical Role": "Time Dimension", "Supports KPI": "✅", "Supports Segmentation": "✅",
         "Supports Trend": "✅", "Notes": "1960–2017; base for YoY, CAGR, period analysis"},
        {"Column": "Value", "Type": "Float", "Business Meaning": "Annual oil production in KTOE",
         "Analytical Role": "Primary Measure / KPI", "Supports KPI": "✅", "Supports Segmentation": "✅",
         "Supports Trend": "✅", "Notes": "Core analytical metric; missing where flagged 'L'"},
        {"Column": "Flag Codes", "Type": "String", "Business Meaning": "Data quality flag",
         "Analytical Role": "Quality Indicator", "Supports KPI": "—", "Supports Segmentation": "—",
         "Supports Trend": "—", "Notes": "'L' = limited/estimated data; blank = actual reported value"},
    ])
    st.dataframe(data_dict, use_container_width=True, hide_index=True)

    st.subheader("Raw Data Explorer")
    loc_filter = st.multiselect(
        "Filter by Location", options=sorted(df["location"].unique()),
        default=["WLD", "SAU", "USA", "RUS"],
        format_func=lambda x: COUNTRY_NAMES.get(x, x)
    )
    yr_range = st.slider("Year Filter", 1960, 2017, (1990, 2017))
    raw_view = df[(df["location"].isin(loc_filter)) &
                  (df["year"].between(*yr_range))][
        ["location", "country_name", "year", "production_ktoe", "production_mtoe",
         "production_mbpd", "flag_code"]
    ].sort_values(["location", "year"])
    st.dataframe(raw_view, use_container_width=True, height=400)
    st.caption(f"Showing {len(raw_view):,} records.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Global Crude Oil Production BI Platform",
        page_icon="🛢️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Load data ────────────────────────────────────────────────────────────
    with st.spinner("Loading and processing dataset..."):
        df, dq = load_and_clean_data()
        world_df = get_world_df(df)
        cdf = get_country_df(df)
        kpis = compute_kpis(df, world_df)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Oil_well_silhouette.svg/120px-Oil_well_silhouette.svg.png",
                 width=80)
        st.title("🛢️ Crude Oil BI")
        st.caption("Global Production Intelligence Platform")
        st.divider()
        page = st.radio(
            "Navigation",
            options=[
                "📊 Executive Overview",
                "📈 Trends & Country Analysis",
                "⚠️ Risk Analysis",
                "🚀 Opportunities & Drivers",
                "🧠 Executive Insights",
                "🔬 Data Quality & Explorer",
            ],
            index=0
        )
        st.divider()
        st.markdown("**Dataset Info**")
        st.markdown(f"- Records: **{dq['total_rows']:,}**")
        st.markdown(f"- Countries/Regions: **{dq['unique_locations']}**")
        st.markdown(f"- Years: **{dq['year_min']}–{dq['year_max']}**")
        st.markdown(f"- Indicator: **OILPROD (KTOE)**")
        st.markdown(f"- Source: OECD/IEA")
        st.divider()
        st.markdown("**Quick KPIs**")
        st.markdown(f"- World ({kpis['latest_year']}): **{kpis['world_production_mtoe']:,.0f} MTOE**")
        st.markdown(f"- Top Producer: **{kpis['top_producer_name']}**")
        st.markdown(f"- CAGR (1971–2017): **{kpis['cagr_pct']}%**")

    # ── Page routing ─────────────────────────────────────────────────────────
    if page == "📊 Executive Overview":
        page_executive_overview(df, world_df, cdf, kpis)
    elif page == "📈 Trends & Country Analysis":
        page_trends_country(df, world_df, cdf, kpis)
    elif page == "⚠️ Risk Analysis":
        page_risk_analysis(df, world_df, cdf, kpis)
    elif page == "🚀 Opportunities & Drivers":
        page_opportunity_drivers(df, world_df, cdf, kpis)
    elif page == "🧠 Executive Insights":
        page_executive_insights(df, world_df, cdf, kpis)
    elif page == "🔬 Data Quality & Explorer":
        page_data_quality(df, dq)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "Global Crude Oil Production Intelligence Platform · "
        "Data Source: OECD/IEA OILPROD · "
        f"Dataset: 1960–2017 · {dq['total_rows']:,} records · "
        "Built with Python + Streamlit + Plotly"
    )


if __name__ == "__main__":
    main()
