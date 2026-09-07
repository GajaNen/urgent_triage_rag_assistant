"""
Streamlit dashboard: submit triage queries against the API, leave feedback,
and monitor the metrics (cost, tokens, latency, answer rate, and user
feedback) collected from every query in the llm_calls table.
"""

import os

import pandas as pd
import requests
import streamlit as st

import db

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Urgent Triage RAG Assistant", layout="wide")
st.title("Urgent Triage RAG Assistant")

query_tab, dashboard_tab = st.tabs(["Ask a question", "Dashboard"])

with query_tab:
    query_text = st.text_area("Describe the patient's presentation:", height=150)
    if st.button("Get ESI assessment") and query_text.strip():
        with st.spinner("Retrieving context and generating assessment..."):
            response = requests.post(f"{API_URL}/query", json={"query": query_text}, timeout=60)
            response.raise_for_status()
            st.session_state["last_result"] = response.json()
            st.session_state["last_query"] = query_text

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        st.markdown(f"**ESI level:** {result['esi_level'] if result['esi_level'] else 'Not determined'}")
        st.write(result["answer"])
        st.caption(f"Latency: {result['latency_seconds']:.2f}s")

        col1, col2 = st.columns(2)
        if col1.button("Helpful"):
            requests.post(f"{API_URL}/feedback", json={
                "query_id": result["query_id"],
                "reaction": "up",
            })
            st.success("Thanks for the feedback!")
        if col2.button("Not helpful"):
            requests.post(f"{API_URL}/feedback", json={
                "query_id": result["query_id"],
                "reaction": "down",
            })
            st.success("Thanks for the feedback!")

with dashboard_tab:
    with db.get_connection() as conn:
        calls = db.load_llm_calls(conn)
        feedback_rows = conn.execute(
            "SELECT reaction, COUNT(*) as count FROM feedback GROUP BY reaction"
        ).fetchall()

    if not calls:
        st.info("No queries logged yet.")
    else:
        df = pd.DataFrame(calls)
        df["created_at"] = pd.to_datetime(df["created_at"])

        st.subheader("1. Cost per query over time")
        st.line_chart(df.set_index("created_at")["cost_usd"])

        st.subheader("2. Token usage per query")
        st.bar_chart(df.set_index("created_at")[["prompt_tokens", "completion_tokens"]])

        st.subheader("3. Response latency distribution")
        st.bar_chart(df["latency_seconds"].value_counts(bins=10).sort_index())

        st.subheader("4. Answer rate by approach")
        st.bar_chart(df.groupby("approach")["answered"].mean())

        st.subheader("5. User feedback")
        feedback_df = pd.DataFrame(feedback_rows, columns=["reaction", "count"])
        if not feedback_df.empty:
            st.bar_chart(feedback_df.set_index("reaction"))
        else:
            st.info("No feedback collected yet.")
