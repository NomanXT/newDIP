"""
Run `streamlit run vis.py` in terminal to start the webpage

Diplomat Activity Database App (Map + Recency Gradient)
======================================================
A **Streamlit** web application to browse, filter, **visualise**, and export U.S.
State-Department diplomat activity records stored in `DIPD_SIM.xlsx`.

Highlights
-----------
* Multi-location aware (`|` delimiter auto-split).
* World map with **red→green** dots: oldest events red, newest green (linear gradient by `Event_ID`).
* Dot radius scales with event frequency.
* Instant CSV export of current filter.

Quick start
-----------
```bash
pip install streamlit pandas openpyxl pydeck geopy
streamlit run vis.py
```
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import pydeck as pdk
import streamlit as st
from geopy.geocoders import Nominatim

################################################################################
# Constants & helpers
################################################################################
DATA_PATH = Path(__file__).with_name("DIPD_SIM.xlsx")
GEOCODER = Nominatim(user_agent="diplomat_app", timeout=5)
DATE_COLS = ["Report_Date", "Travel_Begin", "Travel_End"]
LOCATION_DELIM = "|"

@st.cache_data(show_spinner="Loading data …", persist=True)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def geocode_location(place: str) -> Tuple[float, float] | None:
    if place.lower() in {"domestic", "virtual", ""}:
        return None
    try:
        loc = GEOCODER.geocode(place, language="en")
        if loc:
            return (loc.latitude, loc.longitude)
    except Exception:
        pass
    return None

def split_locations(loc_field: str | List[str]) -> List[str]:
    if pd.isna(loc_field):
        return []
    return [part.strip() for part in str(loc_field).split(LOCATION_DELIM) if part.strip()]

################################################################################
# Page config
################################################################################
st.set_page_config(page_title="Diplomat Activity DB", layout="wide")

st.title("🕊️ Diplomat Activity Database")
st.caption("Interactive explorer — dots fade from red (old) to green (new)")

################################################################################
# Load data
################################################################################
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"❌ `{DATA_PATH.name}` not found.")
    st.stop()

################################################################################
# Sidebar filters
################################################################################
with st.sidebar:
    st.header("🔍 Filters")

    admin_sel = st.multiselect("Administration", sorted(df["Administration"].dropna().unique()))
    lastname_sel = st.multiselect("Last Name", sorted(df["Last_Name"].dropna().unique()))

    all_locations = sorted({loc for cell in df["Location"].dropna() for loc in split_locations(cell)})
    location_sel = st.multiselect("Location", all_locations)

    report_series = df["Report_Date"].dropna()
    min_d = report_series.min().date() if not report_series.empty else None
    max_d = report_series.max().date() if not report_series.empty else None
    date_range = (
        st.date_input("Report Date", (min_d, max_d), min_value=min_d, max_value=max_d)
        if min_d and max_d else None
    )

    text_query = st.text_input("Full-text search in `Text` (case-insensitive)")

################################################################################
# Apply filters
################################################################################
filtered = df.copy()
if admin_sel:
    filtered = filtered[filtered["Administration"].isin(admin_sel)]
if lastname_sel:
    filtered = filtered[filtered["Last_Name"].isin(lastname_sel)]
if location_sel:
    mask = filtered["Location"].apply(lambda cell: any(loc in split_locations(cell) for loc in location_sel))
    filtered = filtered[mask]
if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered = filtered[(filtered["Report_Date"] >= start) & (filtered["Report_Date"] <= end)]
if text_query:
    filtered = filtered[filtered["Text"].str.contains(text_query, case=False, na=False)]

st.subheader(f"Results — {len(filtered):,} event(s)")

################################################################################
# Data table
################################################################################
st.dataframe(filtered, use_container_width=True, height=450)

################################################################################
# Map visualisation with red→green gradient
################################################################################
with st.expander("🗺️ Show world map"):
    unique_places = {loc for cell in filtered["Location"] for loc in split_locations(cell)}
    coords: Dict[str, Tuple[float, float]] = {p: geocode_location(p) for p in unique_places}
    coords = {p: c for p, c in coords.items() if c}

    if not coords:
        st.info("No geocodable locations.")
    else:
        # Frequency per place for radius
        freq = filtered["Location"].apply(split_locations).explode().value_counts().to_dict()

        # Recency (max Event_ID) per place for color
        if "Event_ID" in filtered.columns:
            loc_id_df = (
                filtered[["Location", "Event_ID"]]
                .dropna(subset=["Event_ID"])
                .assign(LocList=lambda d: d["Location"].apply(split_locations))
                .explode("LocList")
            )
            recency = loc_id_df.groupby("LocList")
