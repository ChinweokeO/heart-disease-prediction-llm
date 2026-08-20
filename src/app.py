"""
Streamlit UI for the heart disease risk LLM interface.

Run with:
    streamlit run src/app.py
"""
import os
import sys

import streamlit as st

src_root = os.path.dirname(os.path.abspath(__file__))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from llm_interface import (
    load_model_bundle,
    load_feature_defaults,
    handle_query,
)

st.set_page_config(page_title="Heart Disease Risk Assistant", page_icon="\U0001FAC0")
st.title("Heart Disease Risk Assistant")
st.caption(
    "Describe a patient's clinical measurements in plain English and this tool will "
    "estimate heart disease risk using a trained ML model. This is a statistical "
    "estimate, not a medical diagnosis."
)


@st.cache_resource
def _load_resources():
    model, feature_columns = load_model_bundle()
    defaults = load_feature_defaults()
    return model, feature_columns, defaults


missing_key = None
if os.environ.get("LLM_PROVIDER", "nebius").lower() == "openai":
    if not os.environ.get("OPENAI_API_KEY"):
        missing_key = "OPENAI_API_KEY"
elif not os.environ.get("NEBIUS_API_KEY"):
    missing_key = "NEBIUS_API_KEY"

if missing_key:
    st.error(
        f"Missing environment variable `{missing_key}`. Copy `.env.example` to `.env`, "
        "add your key, and restart the app."
    )
    st.stop()

try:
    model, feature_columns, feature_defaults = _load_resources()
except FileNotFoundError:
    st.error(
        "No trained model found. Run `python src/run_experiments.py` then "
        "`python src/compare_experiments.py` first to train and export a model."
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

user_text = st.chat_input(
    "e.g. I'm a 55-year-old male smoker with chest pain, cholesterol of 240, "
    "resting BP 150, and max heart rate of 130"
)

if user_text:
    st.session_state.history.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                result = handle_query(user_text, model, feature_columns, feature_defaults)
            except Exception as e:
                st.error(f"Something went wrong calling the LLM: {e}")
                st.stop()

        if result["type"] == "clarify":
            st.markdown(result["message"])
            reply = result["message"]

        elif result["type"] == "out_of_scope":
            st.markdown(result["message"])
            reply = result["message"]

        else:
            prob = result["prediction"]["probability"]
            st.markdown(result["explanation"])
            if prob is not None:
                st.progress(min(max(prob, 0.0), 1.0), text=f"Estimated probability: {prob:.0%}")
            if result["assumptions"]:
                with st.expander("Assumptions made"):
                    for a in result["assumptions"]:
                        st.markdown(f"- {a}")
            if result["range_warnings"]:
                with st.expander("Value warnings"):
                    for w in result["range_warnings"]:
                        st.markdown(f"- {w}")
            reply = result["explanation"]

    st.session_state.history.append({"role": "assistant", "content": reply})
