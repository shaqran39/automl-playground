"""Streamlit frontend. Run: streamlit run frontend/streamlit_app.py"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AutoML Playground", layout="wide")
st.title("AutoML Playground")
st.caption("Drop in a CSV. We'll figure out the rest.")

uploaded = st.file_uploader("Upload a CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.session_state["raw_df"] = df
    st.write(f"{df.shape[0]:,} rows, {df.shape[1]} columns")
    st.dataframe(df.head(20))
