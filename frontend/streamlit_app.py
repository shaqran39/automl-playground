"""Streamlit frontend. Run: streamlit run frontend/streamlit_app.py"""

import streamlit as st

from app.core.profiling import read_csv_raw

st.set_page_config(page_title="AutoML Playground", layout="wide")
st.title("AutoML Playground")
st.caption("Drop in a CSV. We'll figure out the rest.")

uploaded = st.file_uploader("Upload a CSV", type="csv")
if uploaded:
    df = read_csv_raw(uploaded)  # raw strings, kept in memory only
    st.session_state["raw_df"] = df
    st.write(f"{df.shape[0]:,} rows, {df.shape[1]} columns")
    st.dataframe(df.head(20))
