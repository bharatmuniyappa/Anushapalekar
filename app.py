# --- Load CSVs from the same folder as app.py (repo root or app folder)
from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np

APP_DIR = Path(__file__).parent.resolve()
ACC_PATH = APP_DIR / "Accidents.csv"
BIK_PATH = APP_DIR / "Bikers.csv"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.replace(r"[^\w_]", "", regex=True)
                  .str.lower())
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
            df["datetime"] = pd.to_datetime(df["_date_parsed"].dt.strftime("%Y-%m-%d") + " " + pd.Series(t, index=df.index).astype(str), errors="coerce")
        else:
            df["datetime"] = df["_date_parsed"]
        df.drop(columns=["_date_parsed"], inplace=True)
    if "datetime" in df.columns:
        df["year"]  = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month
        df["hour"]  = df["datetime"].dt.hour
    return df

def try_merge_acc_bik(acc: pd.DataFrame, bik: pd.DataFrame) -> pd.DataFrame:
    if "accident_index" in acc.columns and "accident_index" in bik.columns:
        t = bik.copy()
        t["_row"] = 1
        agg = t.groupby("accident_index", as_index=False)["_row"].sum().rename(columns={"_row": "casualty_count"})
        out = acc.merge(agg, on="accident_index", how="left")
        out["casualty_count"] = out["casualty_count"].fillna(0).astype(int)
        return out
    return acc.copy()

@st.cache_data
def load_local_or_upload():
    if ACC_PATH.exists() and BIK_PATH.exists():
        acc = pd.read_csv(ACC_PATH, low_memory=False)
        bik = pd.read_csv(BIK_PATH, low_memory=False)
        mode = f"Loaded from repo files: {ACC_PATH.name}, {BIK_PATH.name}"
    else:
        st.info("Couldn’t find local CSVs next to app.py — upload them here.")
        c1, c2 = st.columns(2)
        with c1: acc_file = st.file_uploader("Upload Accidents.csv", type=["csv"])
        with c2: bik_file = st.file_uploader("Upload Bikers.csv", type=["csv"])
        if not (acc_file and bik_file):
            st.stop()
        acc = pd.read_csv(acc_file, low_memory=False)
        bik = pd.read_csv(bik_file, low_memory=False)
        mode = "Loaded from uploaded files"

    acc = normalize_columns(acc)
    bik = normalize_columns(bik)
    acc = build_datetime(acc)
    bik = build_datetime(bik)
    df  = try_merge_acc_bik(acc, bik)

    for c in ["year","month","hour"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, mode

df, mode_msg = load_local_or_upload()
st.caption(mode_msg)
