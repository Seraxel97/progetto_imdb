#!/usr/bin/env python3
"""
Simplified Streamlit dashboard for uploading a CSV and running the analysis pipeline.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from ftfy import fix_text

from scripts.pipeline_runner import run_complete_csv_analysis

st.set_page_config(page_title="Sentiment Analysis", layout="wide")

st.title("Upload & Analyze")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if uploaded and not st.session_state.analysis_done:
    raw_dir = Path("data") / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_csv = raw_dir / f"upload_{timestamp}.csv"
    with open(saved_csv, "wb") as f:
        f.write(uploaded.getbuffer())

    try:
        df = pd.read_csv(saved_csv, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(saved_csv, encoding="latin-1")
    except Exception:
        df = pd.read_csv(saved_csv, encoding="utf-8", errors="replace")

    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns=lambda c: (
        c.replace("review", "text").replace("content", "text")
         .replace("sentiment", "label").replace("class", "label")
    ))
    if "label" in df.columns:
        df["label"] = df["label"].replace({"positive": 1, "negative": 0, "pos": 1, "neg": 0})

    st.session_state.uploaded_df = df

    log_box = st.empty()
    logs = []

    def append_log(text: str):
        logs.append(fix_text(text))
        log_box.text_area("Logs", "".join(logs[-200:]), height=300)

    with st.spinner("Running full pipeline..."):
        results = run_complete_csv_analysis(
            str(saved_csv),
            log_callback=append_log,
        )

    st.session_state.logs = logs

    st.session_state.results = results
    st.session_state.analysis_done = True

if st.session_state.get("analysis_done"):
    df = st.session_state.get("uploaded_df")
    results = st.session_state.get("results")
    logs = st.session_state.get("logs", [])

    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head(200), use_container_width=True, height=300)

    if results and results.get("success"):
        st.success("Analysis completed!")
        final = results.get("final_results", {})
        session_dir = final.get("session_directory")
        summary = final.get("summary", {})
        insights = final.get("insights", [])

        if session_dir:
            plots_dir = Path(session_dir) / "plots"
            reports_dir = Path(session_dir) / "reports"

            if plots_dir.exists():
                for img in sorted(plots_dir.glob("*.png")):
                    st.image(str(img))

            report_json = reports_dir / "evaluation_report.json"
            if report_json.exists():
                st.subheader("Evaluation Report")
                with open(report_json, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
                st.json(report_data)

        st.subheader("Summary Statistics")
        st.json(summary)

        if logs:
            st.subheader("Execution Logs")
            st.text_area("Logs", "".join(logs[-200:]), height=300)

        if insights:
            st.subheader("Insights")
            for ins in insights:
                st.write(f"- {ins}")

        zip_path = final.get("zip_path")
        if zip_path and Path(zip_path).exists():
            with open(zip_path, "rb") as f:
                st.download_button("\U0001F4E5 Download Full Report", f, "report_analysis.zip")
    else:
        st.error(f"Pipeline failed: {results.get('error', 'unknown error')}")
