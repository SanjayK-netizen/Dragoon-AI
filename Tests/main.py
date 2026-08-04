#!/usr/bin/env python3
"""
Dragoon — main entry point (orchestration skeleton).

This file wires the pipeline (STT -> intent -> candidates -> memory ->
agent loop -> TTS) but contains no phase logic itself. Each import below
is phase-gated: if the real module doesn't exist yet, a stub is used
instead, so this file runs today and needs no edits as each phase lands
-- just create the real module and the stub is automatically replaced.

Corrected repo layout (see note in chat: `io/` was renamed to `audio_io/`
because it shadows Python's stdlib `io` module):

dragoon/
  core/
    intent.py        # Phase 1  -> classify_intent, generate_direct_response
    candidates.py     # Phase 2  -> generate_and_score
    memory.py         # Phase 3  -> get_context, update_context
    agent_loop.py      # Phase 4  -> run_agent_loop
    hardening.py      # Phase 6
  tools/
    registry.py         # Phase 4 / 7
  audio_io/
    stt.py               # Phase 0 / 5  -> transcribe_audio
    tts.py               # Phase 0 / 5  -> speak
  data/
    dragoon.db
  logs/
  tests/
  main.py              # this file
"""

import argparse
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

# Confirmed via Phase 0 validation (two independent runs): avg total latency
# ~3.0-3.2s, 100% JSON/tool-call parse rate. qwen3.5:4b was not tested viable
# given this machine's RAM ceiling — do not switch back without re-running
# Phase 0 against it first.
MODEL_NAME = "qwen3.5:2b"


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Phase-gated imports. Each block tries the real module first; if it isn't
# built yet, falls back to a stub that logs a warning and returns something
# safe. No changes needed here as you build each phase.
# --------------------------------------------------------------------------

try:
    from audio_io.stt import transcribe_audio
except ImportError:
    def transcribe_audio():
        logging.getLogger("dragoon").warning(
            "audio_io.stt not implemented yet (Phase 0/5) — stub returns empty text"
        )
        return ""

try:
    from core.intent import classify_intent, generate_direct_response
except ImportError:
    def classify_intent(text):
        logging.getLogger("dragoon").warning(
            "core.intent not implemented yet (Phase 1) — stub classifies everything as conversation"
        )
        return {"intent": "conversation", "raw_text": text, "timestamp": _now_iso()}

    def generate_direct_response(text, context):
        return f"(stub — Phase 1 not implemented) You said: {text}"

try:
    from core.candidates import generate_and_score
except ImportError:
    def generate_and_score(text):
        logging.getLogger("dragoon").warning(
            "core.candidates not implemented yet (Phase 2) — stub auto-executes raw text unchanged"
        )
        return {
            "candidates": [{"text": text, "embedding_score": 0.0, "keyword_score": 0.0, "combined_score": 1.0}],
            "selected_index": 0,
            "action": "auto_execute",
        }

try:
    from core.memory import get_context, update_context
except ImportError:
    def get_context(n=5):
        logging.getLogger("dragoon").warning(
            "core.memory not implemented yet (Phase 3) — stub returns empty context"
        )
        return {}

    def update_context(key, value):
        pass

try:
    from core.agent_loop import run_agent_loop
except ImportError:
    def run_agent_loop(text):
        logging.getLogger("dragoon").warning(
            "core.agent_loop not implemented yet (Phase 4) — stub echoes intended action, executes nothing"
        )
        return f"(stub — Phase 4 not implemented) Would execute: {text}"

try:
    from audio_io.tts import speak
except ImportError:
    def speak(text):
        pass  # no-op until Phase 0/5 wires real TTS; response is still printed to console


def setup_logging(log_level):
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("dragoon")
    logger.setLevel(log_level)

    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    file_handler = RotatingFileHandler("logs/dragoon.log", maxBytes=5 * 1024 * 1024, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Dragoon — local voice assistant")
    parser.add_argument("--config", default="config.json",
                         help="Path to config file. Not yet consumed — each phase module owns its own "
                              "config values (see TRD Section 4) until a real config loader is built.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--no-tts", action="store_true",
                         help="Print responses instead of speaking them (headless testing)")
    parser.add_argument("--text", default=None,
                         help="Bypass STT and inject this text as a single command, then exit (testing)")
    return parser.parse_args()


def process_turn(raw_text, logger):
    """One full pass: classify -> route -> (candidates+agent loop) or (direct response) -> update memory."""
    intent_result = classify_intent(raw_text)
    logger.info(f"intent={intent_result['intent']} text={raw_text!r}")

    if intent_result["intent"] == "command":
        candidate_result = generate_and_score(raw_text)
        if candidate_result["action"] == "disambiguate":
            options = [c["text"] for c in candidate_result["candidates"][:2]]
            logger.info(f"low-confidence routing -> disambiguating between: {options}")
            return "Which did you mean: " + " or ".join(options) + "?"
        selected = candidate_result["candidates"][candidate_result["selected_index"]]
        response = run_agent_loop(selected["text"])
    else:
        context = get_context()
        response = generate_direct_response(raw_text, context)

    update_context("last_command", raw_text)
    return response


def output_response(text, no_tts, logger):
    print(f"Dragoon: {text}")
    logger.info(f"response={text!r}")
    if not no_tts:
        speak(text)


def main():
    args = parse_args()
    logger = setup_logging(args.log_level)
    logger.info(f"Dragoon starting — model={MODEL_NAME}")

    if args.text is not None:
        response = process_turn(args.text, logger)
        output_response(response, args.no_tts, logger)
        return

    logger.info("Entering listen loop (Ctrl+C to exit)")
    try:
        while True:
            raw_text = transcribe_audio()
            if not raw_text:
                continue
            response = process_turn(raw_text, logger)
            output_response(response, args.no_tts, logger)
    except KeyboardInterrupt:
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
