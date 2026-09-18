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
import re
import time
from datetime import datetime, timezone

try:
    import ollama
except ImportError:
    ollama = None

# Keep this in sync with main.py's MODEL_NAME — confirmed via Phase 0.
MODEL_NAME = "qwen3.5:2b"
MODEL_RETRY_ATTEMPTS = 2

logger = logging.getLogger("dragoon")

VALID_INTENTS = {"command", "question", "conversation"}

CLASSIFY_PROMPT_TEMPLATE = (
    "Classify the intent of this text as JSON with a single key \"intent\" whose value is "
    "exactly one of: command, question, conversation.\n"
    "- command: asks the assistant to DO something with a real side effect (set a reminder, "
    "open a file, calculate, control something, send a message, start a timer)\n"
    "- question: asks for factual information or an answer with no side effect\n"
    "- conversation: greetings, small talk, and CREATIVE/ENTERTAINMENT requests — jokes, "
    "stories, poems, pep talks, casual chat, emotional support, social check-ins.\n"
    "IMPORTANT: treat polite wrappers like 'Can you...', 'Could you...', 'Would you...' as a request "
    "for the action that follows, not as a question about ability.\n"
    "Examples:\n"
    "- 'Tell me a joke' -> conversation\n"
    "- 'Can you tell me a joke?' -> conversation\n"
    "- 'Write me a poem' -> conversation\n"
    "- 'How are you doing today?' -> conversation\n"
    "- 'What time is it?' -> question\n"
    "- 'How do I reset my password?' -> question\n"
    "- 'Can you set a reminder for 5pm?' -> command\n"
    "Text: \"{text}\"\n"
    "Respond with JSON only, no other text."
)

QUESTION_WORD_RE = re.compile(
    r"\b(what|who|when|where|why|how|is|are|do|does|did|can|could|would|should|which|whom|whose)\b",
    re.IGNORECASE,
)

CONVERSATION_HINTS = (
    "hello",
    "hi ",
    "hey",
    "good morning",
    "goodnight",
    "how are you",
    "how's it going",
    "what's up",
    "tell me a joke",
    "tell me a story",
    "write me a poem",
    "give me a pep talk",
    "keep me company",
    "i'm bored",
    "thanks",
    "thank you",
    "never mind",
    "just testing you out",
    "i'm feeling stressed",
    "that sounds great",
    "i'm not sure what to do next",
    "that's really helpful",
)

COMMAND_HINTS = (
    "set a reminder",
    "set a timer",
    "start a timer",
    "remind me",
    "open",
    "calculate",
    "add ",
    "send a message",
    "text ",
    "check the weather",
    "create a note",
    "book a meeting",
    "save this",
    "lock",
    "draft an email",
    "delete",
    "remove",
    "turn off",
    "play",
    "move ",
    "schedule",
    "timer",
    "reminder",
)

ARITHMETIC_RE = re.compile(
    r"(?:\b(?:what is|calculate|compute)\b.*)?\d\s*(?:[+\-*/%]|plus|minus|times|divided by)\s*\d",
    re.IGNORECASE,
)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _heuristic_intent(text: str) -> str:
    """Fallback classifier for model failure or transient recovery."""
    value = (text or "").strip().lower()
    if not value:
        return "conversation"

    if any(hint in value for hint in CONVERSATION_HINTS):
        return "conversation"

    if re.search(r"\b(can you|could you|would you)\b.*\b(tell me a joke|tell me a story|write me a poem|give me a pep talk|keep me company)\b", value):
        return "conversation"

    if ARITHMETIC_RE.search(value):
        return "command"

    if re.search(r"\b(set|start|create|schedule)\b.*\b(timer|reminder)\b|\b(timer|reminder)\b.*\b(for|in)\b", value):
        return "command"

    if any(hint in value for hint in COMMAND_HINTS):
        return "command"

    if "?" in value or QUESTION_WORD_RE.search(value):
        return "question"

    return "conversation"


def _parse_model_intent(raw_content: object) -> str:
    """Parse JSON response and normalize valid labels."""
    if raw_content is None:
        raise ValueError("empty model response")

    text = str(raw_content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("model intent payload is not an object")

    candidate = str(parsed.get("intent", "")).strip().lower()
    if candidate in VALID_INTENTS:
        return candidate
    raise ValueError(f"out-of-schema intent: {candidate!r}")


def classify_intent(text: str) -> dict:
    """
    Classify raw_text into one of VALID_INTENTS.

    Retries once on transient model failures and falls back to a conservative
    lexical heuristic when the model is unavailable or returns malformed data.
    """
    if ARITHMETIC_RE.search((text or "").lower()):
        return {
            "intent": "command",
            "raw_text": text,
            "timestamp": _now_iso(),
        }

    prompt = CLASSIFY_PROMPT_TEMPLATE.format(text=text)
    intent = "conversation"

    if ollama is None:
        intent = _heuristic_intent(text)
        logger.warning("ollama is unavailable; using heuristic intent for text=%r -> %s", text, intent)
    else:
        for attempt in range(MODEL_RETRY_ATTEMPTS):
            try:
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    format="json",
                    think=False,
                    options={"temperature": 0.1},
                )
                raw_content = response["message"]["content"]
                intent = _parse_model_intent(raw_content)
                return {
                    "intent": intent,
                    "raw_text": text,
                    "timestamp": _now_iso(),
                }
            except Exception as exc:
                logger.warning(
                    "classify_intent attempt %s/%s failed for text=%r: %s",
                    attempt + 1,
                    MODEL_RETRY_ATTEMPTS,
                    text,
                    exc,
                )
                if attempt + 1 < MODEL_RETRY_ATTEMPTS:
                    time.sleep(0.5)
                    continue
                intent = _heuristic_intent(text)
                logger.warning(
                    "classify_intent: using heuristic fallback for text=%r -> %s",
                    text,
                    intent,
                )
                break

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
        if ollama is None:
            return "Sorry, the language model is unavailable right now."
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