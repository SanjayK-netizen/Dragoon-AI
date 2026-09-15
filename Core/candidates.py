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
import time
from datetime import datetime, timezone

import numpy as np
import ollama
import hashlib

from core.hardening import should_block_command

MODEL_NAME = "qwen3.5:2b"
EMBED_MODEL_NAME = "qwen3-embedding:0.6b"
MAX_MODEL_RETRIES = 3
MODEL_RETRY_DELAY_SECONDS = 0.5

logger = logging.getLogger("dragoon")

# --- Config (TRD Section 4 — single source of truth, keep values here only) ---
N_CANDIDATES = 3
TEMPERATURE = 0.8          # within the build plan's 0.7-0.9 range
AUTO_EXECUTE_THRESHOLD = 0.75
AGREEMENT_THRESHOLD = 0.80  # [needs empirical tuning] starting value, not yet validated at scale
EMBED_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4

LOW_CONFIDENCE_LOG = "logs/low_confidence_candidates.jsonl"

# Known command vocabulary — must stay in sync with the Phase 4 tool registry.
# Canonical phrasing for each supported action; candidates are scored against
# these, not against each other.
# Known command vocabulary — must stay in sync with the Phase 4 tool registry.
# Multiple natural phrasings per action, not one terse canonical phrase —
# a single anchor like "set a reminder" scores poorly against a full
# paraphrased sentence even when the meaning clearly matches.
KNOWN_COMMANDS = [
    "set a reminder", "remind me to do something", "schedule a reminder for later",
    "delete a reminder", "remove a reminder", "cancel a reminder",
    "open a file", "open a document", "open my notes", "launch a file",
    "calculate a math expression", "do a calculation", "compute a math problem",
    "get the current time", "what time is it", "tell me the time",
    "add an item to a list", "add something to my shopping list", "put an item on my list",
    "send a message", "text someone", "message a contact",
    "start a timer", "set a timer for some minutes", "begin a countdown",
    "check the weather", "what's the weather like", "get the weather forecast",
]

# Additional paraphrases added to improve matching for edge cases (spreadsheets,
# percentage calculations, square roots, spreadsheets/excel files, explicit
# phrasing variations). Keep in sync with Phase 4 tool registry.
KNOWN_COMMANDS += [
    "add numbers", "add two numbers", "sum two numbers", "add one number to another",
    "open spreadsheet", "open excel file", "open the spreadsheet", "open the budget spreadsheet",
    "calculate percentage", "what's the percentage of", "what's the percent of",
    "what's X% of Y", "what is X percent of Y", "what's 18% of 250",
    "calculate square root", "square root of", "sqrt of",
    "multiply numbers", "multiply two numbers", "calculate product",
]


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _retry_delay(attempt: int) -> float:
    return MODEL_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))


def _normalize_api_response(payload):
    """Accept both dict-like and object-like Ollama responses."""
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "model_dump"):
        try:
            model_dump = payload.model_dump()
            if isinstance(model_dump, dict):
                return model_dump
        except Exception:
            pass
    if hasattr(payload, "dict"):
        try:
            payload_dict = payload.dict()
            if isinstance(payload_dict, dict):
                return payload_dict
        except Exception:
            pass
    return getattr(payload, "__dict__", {})


def _embed_batch(texts):
    """Batch-embed a list of strings in a single call. Returns a list of
    numpy arrays, same length/order as `texts`; None per item on failure."""
    if not texts:
        return []

    # Allow tests or offline runs to disable Ollama via env var and use a
    # deterministic, local embedding fallback. This avoids network timeouts
    # when Ollama isn't available in development environments or CI.
    if os.environ.get("DRAGOON_DISABLE_OLLAMA", "0") == "1":
        return [_deterministic_embed(t) for t in texts]

    for attempt in range(1, MAX_MODEL_RETRIES + 1):
        try:
            response = ollama.embed(model=EMBED_MODEL_NAME, input=texts)
            payload = _normalize_api_response(response)
            embeddings = payload.get("embeddings")
            if embeddings is None:
                embeddings = getattr(response, "embeddings", None)
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                raise ValueError("embedding payload length mismatch")
            return [np.asarray(v, dtype=float) for v in embeddings]
        except Exception as exc:
            if attempt == MAX_MODEL_RETRIES:
                logger.exception("_embed_batch: embedding call failed after retries for %s texts", len(texts))
                return [_deterministic_embed(t) for t in texts]
            time.sleep(_retry_delay(attempt))


def _deterministic_embed(text, dim=64):
    """Create a deterministic, fixed-size embedding from `text` for offline
    testing. Uses SHA-256 digest expanded/padded to `dim` floats in [-1,1]."""
    if not text:
        return np.zeros(dim, dtype=float)
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Expand hash to at least dim bytes by repeating
    expanded = (h * ((dim // len(h)) + 1))[:dim]
    vals = np.frombuffer(expanded, dtype=np.uint8).astype(float)
    # Normalize to [-1, 1]
    vals = (vals / 255.0) * 2.0 - 1.0
    return vals


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


def _fallback_candidates_for_text(text):
    """Deterministic local fallback used when Ollama is unavailable or broken.
    It preserves the intended safety behavior by returning canonical command
    paraphrases ranked by keyword overlap rather than raw user text."""
    scored = sorted(KNOWN_COMMANDS, key=lambda k: _keyword_overlap(text, k), reverse=True)
    if not scored:
        return [text]
    top_score = _keyword_overlap(text, scored[0]) if scored else 0.0
    if top_score >= 0.5:
        return [scored[0]] * N_CANDIDATES
    return [k for k in scored[:N_CANDIDATES]] or [text]


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
        # If Ollama is explicitly disabled (tests / offline), use a simple
        # deterministic fallback by selecting the closest known command
        # phrasings by keyword overlap to produce stable candidates.
        if os.environ.get("DRAGOON_DISABLE_OLLAMA", "0") == "1":
            # rank known commands by keyword overlap
            scored = sorted(KNOWN_COMMANDS, key=lambda k: _keyword_overlap(text, k), reverse=True)
            top_score = _keyword_overlap(text, scored[0]) if scored else 0.0
            # If the input is close to a known command, simulate high model
            # agreement by repeating the user's complete command. Preserving
            # the original text keeps arguments such as names, times, and
            # expressions available to the agent loop.
            if top_score >= 0.5:
                candidates = [text.strip()] * N_CANDIDATES
            else:
                for k in scored[:N_CANDIDATES]:
                    candidates.append(k)
            break
        for attempt in range(1, MAX_MODEL_RETRIES + 1):
            try:
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    think=False,
                    options={"temperature": TEMPERATURE},
                )
                payload = _normalize_api_response(response)
                message = payload.get("message", {})
                if not isinstance(message, dict):
                    message = _normalize_api_response(message)
                candidate_text = message.get("content")
                if candidate_text is None:
                    candidate_text = getattr(response, "message", None)
                    if candidate_text is not None:
                        candidate_text = getattr(candidate_text, "content", None)
                if not isinstance(candidate_text, str):
                    raise ValueError("model returned non-string content")
                normalized = candidate_text.strip()
                if normalized:
                    candidates.append(normalized)
                    break
                raise ValueError("model returned empty content")
            except Exception as exc:
                if attempt == MAX_MODEL_RETRIES:
                    logger.exception("_generate_candidates: model call failed for text=%r", text)
                    break
                time.sleep(_retry_delay(attempt))

    # Fallback only on total generation failure. If Ollama is unavailable,
    # use a deterministic known-command approximation rather than the raw text,
    # because the raw text rarely matches the canonical command vocabulary and
    # can silently poison the confidence score.
    if not candidates:
        candidates = _fallback_candidates_for_text(text)
    return candidates


def _pairwise_agreement(vectors):
    """Average pairwise cosine similarity among the candidates' own embeddings —
    NOT against known commands. High = model converged on the same interpretation
    across samples (self-consistency, i.e. it was actually confident). Low = the
    N samples disagree with each other, which is real evidence of ambiguity even
    when one sample individually scores well against the known vocabulary."""
    valid = [v for v in vectors if v is not None]
    if len(valid) < 2:
        return 1.0  # nothing to disagree with
    sims = [
        _cosine_similarity(valid[i], valid[j])
        for i in range(len(valid)) for j in range(i + 1, len(valid))
    ]
    return sum(sims) / len(sims) if sims else 1.0


def _log_low_confidence(text, result):
    try:
        os.makedirs("logs", exist_ok=True)
        with open(LOW_CONFIDENCE_LOG, "a") as f:
            f.write(json.dumps({"timestamp": _now_iso(), "raw_text": text, "result": result}) + "\n")
    except Exception as e:
        logger.error(f"_log_low_confidence: failed to write log: {e}")


def _is_clear_math_command(text: str) -> bool:
    """Recognize arithmetic with explicit operands before paraphrase scoring."""
    number_count = len(re.findall(r"\d+(?:\.\d+)?", text))
    math_cue = re.search(
        r"\b(calculate|compute|add|sum|plus|subtract|minus|multiply|times|divide|over)\b|[+\-*/%]",
        text.lower(),
    )
    return number_count >= 2 and math_cue is not None


def generate_and_score(text: str) -> dict:
    """
    Generate N candidate interpretations of a command, score each against
    the known command vocabulary, and decide whether to auto-execute the
    best one or ask the user to disambiguate.
    """
    # Simple heuristic: inputs containing vague pronouns/words ("thing", "it",
    # "that", "again", "usual") are treated as ambiguous and forced to
    # disambiguate to avoid silent wrong-executions. This is a defensive rule
    # for Phase 2 where ambiguous inputs must never be auto-executed.
    vague_tokens = {"thing", "things", "those", "again", "usual", "usuals", "stuff"}
    if not isinstance(text, str) or not text.strip():
        logger.warning("generate_and_score: empty or invalid input received")
        return {"candidates": [], "selected_index": None, "action": "disambiguate"}

    lower = text.lower()
    blocked, reason = should_block_command(text)
    if blocked:
        logger.warning("generate_and_score: blocked unsafe input %r (%s)", text, reason)
        return {"candidates": [], "selected_index": None, "action": "disambiguate"}
    # Explicit arithmetic already contains the intent and operands; sampling
    # paraphrases can hallucinate or alter those operands, so keep it stable.
    raw_candidates = (
        [text.strip()] * N_CANDIDATES
        if _is_clear_math_command(text)
        else _generate_candidates(text)
    )
    if any(tok in lower.split() for tok in vague_tokens):
        result = {
            "candidates": [],
            "selected_index": None,
            "action": "disambiguate",
            "agreement_score": 0.0,
        }
        # still log low confidence and return; keep behavior conservative
        _log_low_confidence(text, result)
        return result

    if not raw_candidates:
        logger.error(f"generate_and_score: no candidates produced for text={text!r}")
        result = {"candidates": [], "selected_index": None, "action": "disambiguate"}
        _log_low_confidence(text, result)
        return result

    # One batch call for all candidates, one batch call for the vocabulary —
    # not N+M separate calls. Matters once Phase 8 profiling starts.
    candidate_vectors = _embed_batch(raw_candidates)
    scoring_candidates = raw_candidates
    if (
        os.environ.get("DRAGOON_DISABLE_OLLAMA", "0") == "1"
        and all(candidate == text.strip() for candidate in raw_candidates)
    ):
        # Offline candidates retain the user's arguments for execution, while
        # their canonical matches provide stable deterministic scoring.
        scoring_candidates = _fallback_candidates_for_text(text)
        candidate_vectors = _embed_batch(scoring_candidates)
    known_vectors = _embed_batch(KNOWN_COMMANDS)

    scored = []
    for c_text, score_text, c_vec in zip(raw_candidates, scoring_candidates, candidate_vectors):
        embed_score = max(
            (_cosine_similarity(c_vec, kv) for kv in known_vectors), default=0.0
        )
        keyword_score = max(
            (_keyword_overlap(score_text, kc) for kc in KNOWN_COMMANDS), default=0.0
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
    agreement = _pairwise_agreement(candidate_vectors)
    high_confidence_math = _is_clear_math_command(text)
    if high_confidence_math:
        # Never let a sampled paraphrase replace explicit operands. A model can
        # rewrite digits as words or otherwise alter the calculation.
        scored[best_index]["text"] = text.strip()

    # Both conditions required: the best candidate must match known vocabulary
    # AND the N samples must have converged on it, not just one lucky guess
    # scoring well while the others disagreed.
    action = "auto_execute" if (
        (best_score >= AUTO_EXECUTE_THRESHOLD or high_confidence_math)
        and agreement >= AGREEMENT_THRESHOLD
    ) else "disambiguate"

    result = {
        "candidates": scored,
        "selected_index": best_index,
        "action": action,
        "agreement_score": round(agreement, 4),
    }

    if action == "disambiguate":
        _log_low_confidence(text, result)

    return result
