import streamlit as st

from app.components.page import page_header
from app.components.styles import apply_enterprise_theme
from app.data import load_automation_candidates, load_cases, load_providers
from src.analytics.assistant_engine import answer_business_question

apply_enterprise_theme()
page_header(
    "Natural-language analytics",
    "Governed analytics assistant",
    "Ask governed operational questions and get answers grounded in the WorkLens dataset.",
)

cases = load_cases()
candidates = load_automation_candidates()
providers = load_providers()

suggestions = [
    "Why are prior authorization cases delayed?",
    "Which activity should we automate first?",
    "Show me the top 10 SLA breach risks.",
    "Which provider causes the most rework?",
]

st.markdown(
    """
    <div class="insight-box">
      <small>GOVERNED ANSWERS</small>
      <p>This local assistant uses validated dashboard data and deterministic
      analytics. It does not send case data to an external model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

question = st.selectbox("Suggested question", suggestions)
custom_question = st.chat_input("Or ask a WorkLens business question")
active_question = custom_question or question

with st.chat_message("user"):
    st.write(active_question)
answer, detail = answer_business_question(active_question, cases, candidates, providers)
with st.chat_message("assistant"):
    st.write(answer)
    if detail is not None:
        st.dataframe(detail, width="stretch", hide_index=True)
