#!/usr/bin/env python3
"""
Simplified Streamlit dashboard for uploading a CSV and running the analysis pipeline.
"""

import streamlit as st
import pandas as pd
import tempfile
from pathlib import Path

from scripts.pipeline_runner import run_complete_csv_analysis

st.set_page_config(page_title="Sentiment Analysis", layout="wide")

st.title("Upload & Analyze")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

df = None

if uploaded:
    try:
        df = pd.read_csv(uploaded, encoding="utf-8", engine="python")
    except Exception:
        df = pd.read_csv(uploaded, encoding="latin-1", engine="python")

    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns=lambda c: (
        c.replace("review", "text").replace("content", "text")
         .replace("sentiment", "label").replace("class", "label")
    ))
    if "label" in df.columns:
        df["label"] = df["label"].replace({"positive": 1, "negative": 0, "pos": 1, "neg": 0})

    st.subheader("Data Preview")
    st.dataframe(df.head(200), use_container_width=True, height=400)

if uploaded and st.button("Start Analysis"):
    with st.spinner("Running pipeline..."):
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_csv = Path(tmp_dir.name) / "data.csv"
        df.to_csv(tmp_csv, index=False)
        results = run_complete_csv_analysis(str(tmp_csv))
        tmp_dir.cleanup()

    if results.get("success"):
        st.success("Analysis completed!")
        summary = results["final_results"].get("summary", {})
        insights = results["final_results"].get("insights", [])

        if "label" in df.columns:
            st.subheader("Label Distribution")
            st.bar_chart(df["label"].value_counts())

        st.subheader("Text Length Distribution")
        st.bar_chart(df["text"].astype(str).str.len())

        st.subheader("Summary Statistics")
        st.json(summary)

        if insights:
            st.subheader("Insights")
            for ins in insights:
                st.write(f"- {ins}")

        zip_path = results["final_results"].get("zip_path")
        if zip_path and Path(zip_path).exists():
            with open(zip_path, "rb") as f:
                st.download_button("\U0001F4E5 Download Complete Report", f, "report_analysis.zip")
    else:
        st.error(f"Pipeline failed: {results.get('error', 'unknown error')}")
