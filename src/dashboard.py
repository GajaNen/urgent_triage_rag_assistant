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
            try:
                response = requests.post(f"{API_URL}/query", json={"query": query_text}, timeout=60)
                response.raise_for_status()
            except requests.RequestException as error:
                detail = error.response.text if error.response is not None else str(error)
                st.error(f"The API could not process this query: {detail}")
            else:
                st.session_state["last_result"] = response.json()
                st.session_state["last_query"] = query_text

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        st.markdown(f"**ESI level:** {result['esi_level'] if result['esi_level'] else 'Not determined'}")
        st.write(result["answer"])
        st.caption(f"Latency: {result['latency_seconds']:.2f}s")

        col1, col2 = st.columns(2)
        if col1.button("Helpful"):
            feedback_response = requests.post(f"{API_URL}/feedback", json={
                "query_id": result["query_id"],
                "reaction": "up",
            })
            if feedback_response.ok:
                st.success("Thanks for the feedback!")
            else:
                st.error(f"Failed to record feedback: {feedback_response.status_code} {feedback_response.text}")
        if col2.button("Not helpful"):
            feedback_response = requests.post(f"{API_URL}/feedback", json={
                "query_id": result["query_id"],
                "reaction": "down",
            })
            if feedback_response.ok:
                st.success("Thanks for the feedback!")
            else:
                st.error(f"Failed to record feedback: {feedback_response.status_code} {feedback_response.text}")

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
        df = df.sort_values("created_at").reset_index(drop=True)
        df["query_label"] = [f"Query {index + 1}: {query[:45]}" for index, query in enumerate(df["query"])]

        st.subheader("1. Cost per query over time")
        cost_chart = df[["created_at", "cost_usd"]].rename(columns={"created_at": "Query time", "cost_usd": "Cost (USD)"})
        st.line_chart(cost_chart, x="Query time", y="Cost (USD)")

        st.subheader("2. Token usage per query")
        token_chart = df[["query_label", "prompt_tokens", "completion_tokens"]].rename(columns={
            "query_label": "Query",
            "prompt_tokens": "Prompt tokens",
            "completion_tokens": "Completion tokens",
        })
        st.bar_chart(token_chart, x="Query", y=["Prompt tokens", "Completion tokens"])

        st.subheader("3. Response latency distribution")
        latency_counts = df["latency_seconds"].value_counts(bins=10).sort_index()
        latency_chart = pd.DataFrame({
            "Latency range (seconds)": [str(interval) for interval in latency_counts.index],
            "Number of queries": latency_counts.to_numpy(),
        })
        st.bar_chart(latency_chart, x="Latency range (seconds)", y="Number of queries")

        st.subheader("4. Answer rate by approach")
        answer_rate = df.groupby("approach", as_index=False)["answered"].mean()
        answer_rate = answer_rate.rename(columns={"approach": "Retrieval approach", "answered": "Answer rate"})
        st.bar_chart(answer_rate, x="Retrieval approach", y="Answer rate")

        st.subheader("5. User feedback")
        feedback_df = pd.DataFrame(feedback_rows, columns=["reaction", "count"])
        if not feedback_df.empty:
            feedback_df = feedback_df.rename(columns={"reaction": "Feedback reaction", "count": "Number of reactions"})
            st.bar_chart(feedback_df, x="Feedback reaction", y="Number of reactions")
        else:
            st.info("No feedback collected yet.")
