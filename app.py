# =============================================================================
# Streamlit App: GB Bicycle Accidents Dashboard (repo-local CSVs + upload fallback)
# Place Accidents.csv and Bikers.csv in the SAME FOLDER as this app.py
# If not found, the app will ask you to upload them.
# =============================================================================

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --- Safe import for Plotly ---
try:
    import plotly.express as px
except Exception as _e:
    st.error(
        "Plotly is not installed in this environment.\n\n"
        "Fix: add `plotly` to requirements.txt or run `pip install plotly` locally.\n\n"
        f"Original import error: {_e}"
    )
    st.stop()

st.set_page_config(page_title="GB Bicycle Accidents Dashboard", layout="wide")

st.title("🚲 Great Britain Bicycle Accidents (1979–2018)")
st.caption("Reads CSVs from the same folder as app.py. If not present, you can upload them.")

# --------------------------------------------------------------------------------
# File locations (relative to this file)
# --------------------------------------------------------------------------------
APP_DIR = Path(__file__).parent.resolve()
ACC_PATH = APP_DIR / "Accidents.csv"
BIK_PATH = APP_DIR / "Bikers.csv"

# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w_]", "", regex=True)
        .str.lower()
    )
    return df

def build_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_col = next((c for c in ["date", "accident_date"] if c in df.columns), None)
    time_col = next((c for c in ["time", "accident_time"] if c in df.columns), None)

    if date_col:
        df["_date_parsed"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        if time_col and time_col in df.columns:
            t = df[time_col].astype(str).str.strip()
            t = np.where(pd.Series(t).str.contains(":", na=False), t,
                         pd.Series(t).str.replace(r"^(\d{1,2})(\d{2})$", r"\1:\2", regex=True))
            df["datetime"] = pd.to_datetime(
                df["_date_parsed"].dt.strftime("%Y-%m-%d") + " " + pd.Series(t, index=df.index).astype(str),
                errors="coerce"
            )
        else:
            df["datetime"] = df["_date_parsed"]
        df.drop(columns=["_date_parsed"], inplace=True)

    if "datetime" in df.columns:
        df["year"]  = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month
        df["hour"]  = df["datetime"].dt.hour
        df["day_name"] = df["datetime"].dt.day_name()
    return df

def try_merge_acc_bik(acc: pd.DataFrame, bik: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal merge to add casualty_count if both tables contain 'accident_index'.
    """
    if "accident_index" in acc.columns and "accident_index" in bik.columns:
        t = bik.copy()
        t["_row"] = 1
        agg = t.groupby("accident_index", as_index=False)["_row"].sum().rename(columns={"_row": "casualty_count"})
        merged = acc.merge(agg, on="accident_index", how="left")
        merged["casualty_count"] = merged["casualty_count"].fillna(0).astype(int)
        return merged
    return acc.copy()

@st.cache_data
def load_local_or_upload() -> tuple[pd.DataFrame, str]:
    """
    If Accidents.csv and Bikers.csv exist next to app.py, load them.
    Otherwise, prompt for upload. Returns (df, mode_message).
    """
    if ACC_PATH.exists() and BIK_PATH.exists():
        acc = pd.read_csv(ACC_PATH, low_memory=False)
        bik = pd.read_csv(BIK_PATH, low_memory=False)
        mode = f"Loaded from repo files: {ACC_PATH.name}, {BIK_PATH.name}"
    else:
        st.info("Local CSVs not found next to app.py — upload the two files below.")
        c1, c2 = st.columns(2)
        with c1:
            acc_file = st.file_uploader("Upload Accidents.csv", type=["csv"], key="acc_upl")
        with c2:
            bik_file = st.file_uploader("Upload Bikers.csv", type=["csv"], key="bik_upl")
        if not (acc_file and bik_file):
            st.stop()
        acc = pd.read_csv(acc_file, low_memory=False)
        bik = pd.read_csv(bik_file, low_memory=False)
        mode = "Loaded from uploaded files"

    # Normalize + enrich
    acc = normalize_columns(acc)
    bik = normalize_columns(bik)
    acc = build_datetime(acc)
    bik = build_datetime(bik)
    df  = try_merge_acc_bik(acc, bik)

    # Defensive coercions
    for c in ["year", "month", "hour"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep common categorical-like as strings for filters
    for c in ["accident_severity", "road_type", "light_conditions", "weather_conditions",
              "urban_or_rural_area", "local_authority_district", "police_force"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df, mode

# Load data
df, mode_msg = load_local_or_upload()
st.caption(mode_msg)

# --------------------------------------------------------------------------------
# Sidebar: options & filters
# --------------------------------------------------------------------------------
st.sidebar.header("Filters")

# Optional sampling for very large datasets (speeds up charts)
use_sample = st.sidebar.checkbox("Use sample for speed (recommended for very large files)", value=False)
sample_rows = st.sidebar.number_input("Sample size", min_value=50_000, max_value=900_000, value=250_000, step=50_000)
if use_sample and len(df) > sample_rows:
    df = df.sample(int(sample_rows), random_state=42).copy()
    st.caption(f"Sampling enabled: using {len(df):,} rows")

# Year filter
if "year" in df.columns and df["year"].notna().any():
    y_min = int(df["year"].min())
    y_max = int(df["year"].max())
    year_range = st.sidebar.slider("Year range", min_value=y_min, max_value=y_max, value=(y_min, y_max))
else:
    year_range = None

# Severity filter
sev_col = "accident_severity" if "accident_severity" in df.columns else None
if sev_col:
    sev_opts = ["All"] + sorted([s for s in df[sev_col].dropna().unique()])
    sev_pick = st.sidebar.selectbox("Accident severity", sev_opts, index=0)
else:
    sev_pick = "All"

def multi(col_name: str, label: str):
    if col_name in df.columns:
        opts = sorted([s for s in df[col_name].dropna().unique()])
        return st.sidebar.multiselect(label, ["(All)"] + opts, default=["(All)"])
    return None

road_pick    = multi("road_type", "Road type (multi)")
light_pick   = multi("light_conditions", "Light conditions (multi)")
weather_pick = multi("weather_conditions", "Weather conditions (multi)")
area_pick    = multi("urban_or_rural_area", "Urban/Rural (multi)")

# Apply filters
mask = pd.Series(True, index=df.index)

if year_range and "year" in df.columns:
    mask &= (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
if sev_col and sev_pick != "All":
    mask &= df[sev_col].astype(str) == str(sev_pick)

def apply_multi(mask, col_name, picks):
    if col_name not in df.columns or picks is None:
        return mask
    if "(All)" in picks:
        return mask
    return mask & df[col_name].isin(picks)

mask = apply_multi(mask, "road_type", road_pick)
mask = apply_multi(mask, "light_conditions", light_pick)
mask = apply_multi(mask, "weather_conditions", weather_pick)
mask = apply_multi(mask, "urban_or_rural_area", area_pick)

dff = df[mask].copy()

# --------------------------------------------------------------------------------
# KPI row
# --------------------------------------------------------------------------------
total_acc = len(dff)
avg_cas = dff.get("casualty_count", pd.Series([np.nan])).mean()
avg_cas_str = f"{avg_cas:.2f}" if isinstance(avg_cas, (int, float, np.floating)) and not math.isnan(avg_cas) else "N/A"

sev_share = {}
if sev_col and sev_col in dff.columns:
    sev_share = dff[sev_col].value_counts(normalize=True).mul(100).round(1).to_dict()

c1, c2, c3 = st.columns(3)
c1.metric("Total accidents (filtered)", f"{total_acc:,}")
c2.metric("% by severity", json.dumps(sev_share) if sev_share else "N/A")
c3.metric("Avg casualties per accident", avg_cas_str)

# --------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------
if "year" in dff.columns and dff["year"].notna().any():
    y = dff["year"].value_counts().sort_index().reset_index()
    y.columns = ["year", "accidents"]
    st.plotly_chart(px.line(y, x="year", y="accidents", title="Accidents per Year"), use_container_width=True)

if "month" in dff.columns and dff["month"].notna().any():
    m = dff["month"].value_counts().sort_index().reset_index()
    m.columns = ["month", "accidents"]
    st.plotly_chart(px.bar(m, x="month", y="accidents", title="Accidents by Month (Aggregated)"), use_container_width=True)

if "hour" in dff.columns and dff["hour"].notna().any():
    h = dff["hour"].value_counts().sort_index().reset_index()
    h.columns = ["hour", "accidents"]
    st.plotly_chart(px.line(h, x="hour", y="accidents", title="Accidents by Hour of Day"), use_container_width=True)

if "road_type" in dff.columns:
    rt = dff["road_type"].value_counts().head(10).reset_index()
    rt.columns = ["road_type", "count"]
    st.plotly_chart(px.bar(rt, x="count", y="road_type", orientation="h", title="Top Road Types (Top 10)"), use_container_width=True)

if sev_col and sev_col in dff.columns:
    sev_counts = dff[sev_col].value_counts().reset_index()
    sev_counts.columns = [sev_col, "count"]
    st.plotly_chart(px.pie(sev_counts, names=sev_col, values="count", hole=0.45, title="Accident Severity Share"), use_container_width=True)

if "weather_conditions" in dff.columns and "light_conditions" in dff.columns:
    combo = dff.groupby(["weather_conditions", "light_conditions"]).size().reset_index(name="accidents")
    top10 = combo.sort_values("accidents", ascending=False).head(10)
    st.plotly_chart(
        px.bar(
            top10,
            x="accidents",
            y="weather_conditions",
            color="light_conditions",
            orientation="h",
            title="Top Weather × Light Condition Combinations (Top 10)"
        ),
        use_container_width=True
    )

# Optional map (sampled)
lat_col = next((c for c in ["latitude", "lat"] if c in dff.columns), None)
lon_col = next((c for c in ["longitude", "lon", "lng"] if c in dff.columns), None)
if lat_col and lon_col:
    sample_n = min(len(dff), 10_000)
    map_df = dff[[lat_col, lon_col, sev_col] if sev_col else [lat_col, lon_col]].dropna()
    if len(map_df) > sample_n:
        map_df = map_df.sample(sample_n, random_state=42)
    fig_map = px.scatter_mapbox(
        map_df,
        lat=lat_col,
        lon=lon_col,
        color=sev_col if sev_col and sev_col in map_df.columns else None,
        zoom=4,
        height=520,
        title="Accident Locations (sampled)"
    )
    fig_map.update_layout(mapbox_style="carto-positron")
    st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")
st.caption("© Your Name — Streamlit dashboard. Place CSVs next to app.py or upload them via the UI.")
