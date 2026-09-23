"""Streamlit interface for the working Dragoon assistant pipeline."""

import logging

import streamlit as st

from IO.tts import speak
from IO.stt import transcribe_audio
from app import process_turn, setup_logging


st.set_page_config(
    page_title="Dragoon AI",
    page_icon="🐉",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #2b1521 0%, #101a38 45%, #070a12 100%);
        color: #eef3ff;
    }
    .hero {
        border: 1px solid #d13b4f;
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #202f61, #461d32);
    }
    .hero h1 { color: #ff6b72; margin-bottom: 0.25rem; }
    .hero p { color: #b7c8e8; margin-bottom: 0; }
    div.stButton > button[kind="primary"] {
        background: #c9344b;
        border-color: #ff6570;
    }
    div.stButton > button[kind="secondary"] {
        border-color: #3f7ee8;
        color: #b9d2ff;
    }
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

listen = st.button("Listen from microphone")

if listen:
    with st.spinner("Listening..."):
        command = transcribe_audio()
    submitted = bool(command.strip())
    speak_response = True

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
