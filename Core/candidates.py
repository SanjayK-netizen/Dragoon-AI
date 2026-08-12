"""
Dragoon — core/candidates.py (Phase 2)

Exports:
  generate_and_score(text) -> dict   matching TRD Section 3.2:
    {
      "candidates": [{"text": str, "embedding_score": float,
                       "keyword_score": float, "combined_score": float}, ...],
      "selected_index": int | None,
      "action": "auto_execute" | "disambiguate"
    }

Called only for command-type input (after core.intent classifies it).
Never raises. On total failure (no candidates producible at all), forces
"disambiguate" rather than guessing — silent wrong-execution is the one
outcome Phase 2's exit criteria explicitly forbids, so failure must never
look like confidence.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone

import numpy as np
import ollama

MODEL_NAME = "qwen3.5:2b"
EMBED_MODEL_NAME = "qwen3-embedding:0.6b"

logger = logging.getLogger("dragoon")

# --- Config (TRD Section 4 — single source of truth, keep values here only) ---
N_CANDIDATES = 3
TEMPERATURE = 0.8          # within the build plan's 0.7-0.9 range
AUTO_EXECUTE_THRESHOLD = 0.7
EMBED_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4

LOW_CONFIDENCE_LOG = "logs/low_confidence_candidates.jsonl"

# Very short, referential commands are too vague to safely auto-execute.
# These are the exact failure modes Phase 2 must prevent.
VAGUE_COMMAND_PATTERN = re.compile(
    r"\b(it|that|this|one|them|those|these|thing|usual)\b",
    flags=re.IGNORECASE,
)

# Known command vocabulary — must stay in sync with the Phase 4 tool registry.
# Canonical phrasing for each supported action; candidates are scored against
# these, not against each other.
# Multiple natural phrasings per action, not one terse canonical phrase —
# a single anchor like "set a reminder" scores poorly against a full
# paraphrased sentence even when the meaning clearly matches.
KNOWN_COMMANDS = [
    "set a reminder", "remind me to do something", "remind me to call someone", "schedule a reminder for later",
    "delete a reminder", "remove a reminder", "cancel a reminder",
    "open a file", "open a document", "open my notes", "open a spreadsheet", "open a spreadsheet file", "open a file called something", "launch a file",
    "calculate a math expression", "do a calculation", "compute a math problem", "calculate a percentage", "compute a percentage", "calculate a square root", "compute a square root", "multiply numbers",
    "get the current time", "what time is it", "tell me the time",
    "add an item to a list", "add something to my shopping list", "put an item on my list", "add item to my list", "add to my list", "add an item to the grocery list",
    "send a message", "send a text", "send a text message", "text someone", "text a contact", "text a person", "message a contact",
    "start a timer", "set a timer for some minutes", "begin a countdown",
    "check the weather", "what's the weather like", "get the weather forecast",
]


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _is_vague_command(text):
    lower = text.strip().lower()
    tokens = lower.split()
    if len(tokens) <= 5 and VAGUE_COMMAND_PATTERN.search(lower):
        return True
    if "the thing" in lower or "that thing" in lower or "this thing" in lower or "the usual" in lower:
        return True
    if "reminder thing" in lower:
        return True
    return False


def _embed_batch(texts):
    """Batch-embed a list of strings in a single call. Returns a list of
    numpy arrays, same length/order as `texts`; None per item on failure."""
    if not texts:
        return []
    try:
        response = ollama.embed(model=EMBED_MODEL_NAME, input=texts)
        return [np.array(v) for v in response["embeddings"]]
    except Exception as e:
        logger.error(f"_embed_batch: embedding call failed for {len(texts)} texts: {e}")
        return [None] * len(texts)


def _cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _keyword_overlap(candidate_text, known_command):
    c_words = set(candidate_text.lower().split())
    k_words = set(known_command.lower().split())
    if not k_words:
        return 0.0
    return len(c_words & k_words) / len(k_words)


def _generate_candidates(text):
    """Sample N_CANDIDATES interpretations of a command via temperature sampling."""
    candidates = []
    prompt = (
        "Rephrase this user command as a SHORT canonical action phrase — 3-6 words, "
        "not a full sentence. Match the style of: \"set a reminder\", \"open a file\", "
        "\"start a timer\". No preamble, no punctuation beyond the phrase itself.\n"
        f"Command: \"{text}\""
    )
    for _ in range(N_CANDIDATES):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                think=False,
                options={"temperature": TEMPERATURE},
            )
            candidate_text = response["message"]["content"].strip()
            if candidate_text:
                candidates.append(candidate_text)
        except Exception as e:
            logger.error(f"_generate_candidates: model call failed for text={text!r}: {e}")

    # Original raw text is always included as a fallback candidate, so
    # scoring never runs on an empty list even if every generation call failed.
    if text not in candidates:
        candidates.append(text)
    return candidates


def _log_low_confidence(text, result):
    try:
        os.makedirs("logs", exist_ok=True)
        with open(LOW_CONFIDENCE_LOG, "a") as f:
            f.write(json.dumps({"timestamp": _now_iso(), "raw_text": text, "result": result}) + "\n")
    except Exception as e:
        logger.error(f"_log_low_confidence: failed to write log: {e}")


def generate_and_score(text: str) -> dict:
    """
    Generate N candidate interpretations of a command, score each against
    the known command vocabulary, and decide whether to auto-execute the
    best one or ask the user to disambiguate.
    """
    raw_candidates = _generate_candidates(text)

    if not raw_candidates:
        logger.error(f"generate_and_score: no candidates produced for text={text!r}")
        result = {"candidates": [], "selected_index": None, "action": "disambiguate"}
        _log_low_confidence(text, result)
        return result

    # One batch call for all candidates, one batch call for the vocabulary —
    # not N+M separate calls. Matters once Phase 8 profiling starts.
    candidate_vectors = _embed_batch(raw_candidates)
    known_vectors = _embed_batch(KNOWN_COMMANDS)

    scored = []
    for c_text, c_vec in zip(raw_candidates, candidate_vectors):
        embed_score = max(
            (_cosine_similarity(c_vec, kv) for kv in known_vectors), default=0.0
        )
        keyword_score = max(
            (_keyword_overlap(c_text, kc) for kc in KNOWN_COMMANDS), default=0.0
        )
        combined = EMBED_WEIGHT * embed_score + KEYWORD_WEIGHT * keyword_score
        scored.append({
            "text": c_text,
            "embedding_score": round(embed_score, 4),
            "keyword_score": round(keyword_score, 4),
            "combined_score": round(combined, 4),
        })

    best_index = max(range(len(scored)), key=lambda i: scored[i]["combined_score"])
    best_score = scored[best_index]["combined_score"]
    if _is_vague_command(text):
        action = "disambiguate"
    else:
        action = "auto_execute" if best_score >= AUTO_EXECUTE_THRESHOLD else "disambiguate"

    result = {"candidates": scored, "selected_index": best_index, "action": action}

    if action == "disambiguate":
        _log_low_confidence(text, result)

    return result
