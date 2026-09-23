"""Streamlit interface for the working Dragoon assistant pipeline."""

import logging

import streamlit as st

from IO.tts import speak
from Tests.main import process_turn, setup_logging


st.set_page_config(
    page_title="Dragoon AI",
    page_icon="🐉",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #18243d 0%, #0b1020 55%, #070a12 100%);
        color: #eef3ff;
    }
    .hero {
        border: 1px solid #31466f;
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #18294c, #10182d);
    }
    .hero h1 { color: #8dd6ff; margin-bottom: 0.25rem; }
    .hero p { color: #b7c8e8; margin-bottom: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_logger() -> logging.Logger:
    return setup_logging("INFO")


st.markdown(
    """
    <div class="hero">
      <h1>🐉 Dragoon AI</h1>
      <p>Local assistant pipeline with safe tool execution and voice responses.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("command_form", clear_on_submit=True):
    command = st.text_input(
        "Command",
        placeholder="Calculate 12 + 7",
        help="Enter a command for the same pipeline used by app.py.",
    )
    speak_response = st.checkbox("Speak response", value=True)
    submitted = st.form_submit_button("Run command", type="primary")

if submitted:
    if not command.strip():
        st.warning("Enter a command first.")
    else:
        logger = get_logger()
        with st.spinner("Dragoon is processing..."):
            response = process_turn(command, logger)
        st.session_state.history.append((command, response))
        if speak_response:
            speak(response)

for user_text, response in reversed(st.session_state.history):
    st.markdown(f"**You:** {user_text}")
    st.info(response)
