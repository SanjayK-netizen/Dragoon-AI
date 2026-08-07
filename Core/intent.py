"""
Dragoon — core/intent.py (Phase 1)

Exports:
  classify_intent(text) -> dict   matching TRD Section 3.1:
    {"intent": "command|question|conversation", "raw_text": str, "timestamp": ISO8601}
  generate_direct_response(text, context) -> str
    Used for the question/conversation branch in main.py's process_turn().

Never raises. Any model/parse failure falls back to a safe default rather
than propagating an exception up into main.py's orchestration loop.
"""

import json
import logging
from datetime import datetime, timezone

import ollama

# Keep this in sync with main.py's MODEL_NAME — confirmed via Phase 0
# (5 passing runs: ~1.1-3.7s first-token, ~3.0-6.1s total, 7/7 JSON).
MODEL_NAME = "qwen3.5:2b"

logger = logging.getLogger("dragoon")

VALID_INTENTS = {"command", "question", "conversation"}

CLASSIFY_PROMPT_TEMPLATE = (
    "Classify the intent of this text as JSON with a single key \"intent\" whose value is "
    "exactly one of: command, question, conversation.\n"
    "- command: asks the assistant to DO something (set a reminder, open a file, calculate, "
    "control something)\n"
    "- question: asks for information, no action required\n"
    "- conversation: greetings, small talk, anything that is neither of the above\n"
    "Text: \"{text}\"\n"
    "Respond with JSON only, no other text."
)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def classify_intent(text: str) -> dict:
    """
    Classify raw_text into one of VALID_INTENTS.

    Falls back to "conversation" on any parse failure, model error, or an
    out-of-schema label — main.py's routing assumes intent is always one
    of the three valid values, so this must never return anything else.
    """
    prompt = CLASSIFY_PROMPT_TEMPLATE.format(text=text)
    intent = "conversation"  # safe default if anything below fails

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            think=False,  # Phase 0 finding: leaving this on cost 10-100x latency
            options={"temperature": 0.1},  # low temp — classification should be stable, not creative
        )
        raw_content = response["message"]["content"]
        parsed = json.loads(raw_content)
        candidate = str(parsed.get("intent", "")).strip().lower()

        if candidate in VALID_INTENTS:
            intent = candidate
        else:
            logger.warning(
                f"classify_intent: model returned out-of-schema intent {candidate!r} "
                f"for text={text!r} — defaulting to conversation"
            )

    except json.JSONDecodeError as e:
        logger.error(f"classify_intent: unparseable JSON for text={text!r}: {e}")
    except Exception as e:
        logger.error(f"classify_intent: model call failed for text={text!r}: {e}")

    return {
        "intent": intent,
        "raw_text": text,
        "timestamp": _now_iso(),
    }


def generate_direct_response(text: str, context: dict) -> str:
    """
    Direct response for question/conversation intents.
    `context` comes from core.memory.get_context() (Phase 3) — empty dict
    is expected and fine until that module exists.
    """
    context_block = f"\nRelevant context: {json.dumps(context)}\n" if context else ""
    prompt = (
        "You are Dragoon, a local voice assistant. Respond naturally and briefly "
        f"(1-3 sentences — this gets spoken aloud).{context_block}\n"
        f"User: {text}"
    )

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            think=False,
            options={"temperature": 0.7},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        logger.error(f"generate_direct_response failed for text={text!r}: {e}")
        return "Sorry, I ran into a problem answering that."
