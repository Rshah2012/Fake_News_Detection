from __future__ import annotations

import os
import streamlit as st

from search_google import google_custom_search

# Prevent tokenizer parallelism warnings / occasional weirdness on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="Fake News Checker", page_icon="🧪", layout="wide")

# Render UI immediately
st.title("🧪 Fake / Real headline checker (NLI + Google Search)")
st.write("✅ UI loaded. Model will load only when you click **Check**.")
st.caption(
    "Pipeline: Google Custom Search → retrieve snippets → MNLI (entail/contradict/neutral) "
    "to bucket into Supports / Contradicts / Unclear."
)

# --------- Secrets (Streamlit Cloud: set these in Secrets) ----------
API_KEY = st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else ""
CX = st.secrets["GOOGLE_CX"] if "GOOGLE_CX" in st.secrets else ""

with st.sidebar:
    st.header("Settings")
    num_results = st.slider("Google results", 5, 20, 10)
    top_k = st.slider("Show top-k per bucket", 3, 15, 10)
    tau_entail = st.slider("Entail threshold", 0.50, 0.95, 0.65, 0.01)
    tau_contra = st.slider("Contradict threshold", 0.50, 0.95, 0.65, 0.01)
    show_neutral = st.checkbox("Show 'Unclear / Neutral'", value=True)

    st.divider()
    st.write("Secrets required:")
    st.code("GOOGLE_API_KEY\nGOOGLE_CX", language="text")

# --------- Cache model so it loads once ----------
@st.cache_resource
def load_checker():
    from nli_checker import NLIChecker
    return NLIChecker()

headline = st.text_input(
    "Enter a news headline",
    placeholder="e.g., India beat South Africa in the first ODI"
)

run = st.button("Check", type="primary", disabled=not headline.strip())

if run:
    if not API_KEY or not CX:
        st.error("Missing GOOGLE_API_KEY / GOOGLE_CX in Streamlit secrets.")
        st.stop()

    # Load model lazily (prevents blank screen on CPU first load)
    with st.spinner("Loading NLI model (first time can take a while on CPU)..."):
        checker = load_checker()

    with st.spinner("Searching web snippets..."):
        items = google_custom_search(headline, api_key=API_KEY, cx=CX, num=num_results)

    if not items:
        st.warning("No search results returned. Verdict: UNCLEAR (no evidence found).")
        st.stop()

    with st.spinner("Running NLI checks..."):
        results = []
        for it in items:
            if it.combined_text.strip():
                r = checker.decision_bidir(
                    it.combined_text,
                    headline,
                    tau_entail=tau_entail,
                    tau_contra=tau_contra,
                )
                results.append((it, r))

        supports = [(it, r) for it, r in results if r.verdict == "MATCH_ENTAILS"]
        contradicts = [(it, r) for it, r in results if r.verdict == "MISMATCH_CONTRADICTS"]
        unclear = [(it, r) for it, r in results if r.verdict == "UNCLEAR_NEUTRAL"]

        supports.sort(key=lambda x: x[1].entailment, reverse=True)
        contradicts.sort(key=lambda x: x[1].contradiction, reverse=True)
        unclear.sort(key=lambda x: x[1].neutral, reverse=True)

        supports = supports[:top_k]
        contradicts = contradicts[:top_k]
        unclear = unclear[:top_k]

    # --------- High-level verdict (simple rule) ----------
    if supports and not contradicts:
        overall = "LIKELY REAL (supported by retrieved snippets)"
        badge = "✅"
    elif contradicts and not supports:
        overall = "LIKELY FAKE / MISLEADING (contradicted by retrieved snippets)"
        badge = "❌"
    else:
        overall = "UNCLEAR (mixed, weak, or neutral evidence)"
        badge = "⚠️"

    st.subheader(f"{badge} {overall}")
    st.write("**Input headline:**", headline)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Supports (Entails)")
        if not supports:
            st.info("No strong supports found.")
        for it, r in supports:
            with st.expander(f"{r.entailment:.3f} entail | {it.title}", expanded=False):
                st.write(it.snippet)
                if it.link:
                    st.link_button("Open source", it.link)
                st.code(r.details, language="json")

    with col2:
        st.markdown("### ❌ Contradicts")
        if not contradicts:
            st.info("No strong contradictions found.")
        for it, r in contradicts:
            with st.expander(f"{r.contradiction:.3f} contra | {it.title}", expanded=False):
                st.write(it.snippet)
                if it.link:
                    st.link_button("Open source", it.link)
                st.code(r.details, language="json")

    if show_neutral:
        st.markdown("### ➖ Unclear / Neutral")
        if not unclear:
            st.info("No neutral items to show.")
        for it, r in unclear:
            with st.expander(f"{r.neutral:.3f} neutral | {it.title}", expanded=False):
                st.write(it.snippet)
                if it.link:
                    st.link_button("Open source", it.link)
                st.code(r.details, language="json")