#!/usr/bin/env python3
"""
Dragoon Phase 0 — Environment & Model Validation
Run: python phase0_validation.py [model_name]
Default model: qwen3.5:4b

Tests 10 prompts (plain Q&A, JSON extraction, tool-call style), measures
first-token and total latency separately, and checks JSON/tool-call output
actually parses. Implements the Phase 0 decision gate from the build plan.
"""

import sys
import time
import json

import ollama

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5:4b"

TEST_PROMPTS = [
    {"type": "qa", "prompt": "What is the capital of France? Answer in one sentence."},
    {"type": "qa", "prompt": "Explain what a SQLite database is in one sentence."},
    {"type": "qa", "prompt": "What year did the first moon landing happen?"},
    {"type": "json", "prompt": (
        "Classify the intent of this text as JSON with a single key \"intent\" whose value is "
        "exactly one of: command, question, conversation. Text: \"Set a reminder for 5pm.\" "
        "Respond with JSON only, no other text."
    )},
    {"type": "json", "prompt": (
        "Classify the intent of this text as JSON with a single key \"intent\" whose value is "
        "exactly one of: command, question, conversation. Text: \"What's the weather like?\" "
        "Respond with JSON only, no other text."
    )},
    {"type": "json", "prompt": (
        "Classify the intent of this text as JSON with a single key \"intent\" whose value is "
        "exactly one of: command, question, conversation. Text: \"Tell me a joke.\" "
        "Respond with JSON only, no other text."
    )},
    {"type": "tool", "prompt": (
        "You have a tool called \"set_reminder\" with args {time: string, message: string}. "
        "The user said: \"Remind me to call mom at 6pm.\" "
        "Respond ONLY with JSON in the form {\"tool_name\": \"...\", \"args\": {...}}."
    )},
    {"type": "tool", "prompt": (
        "You have a tool called \"get_time\" with no args. The user said: \"What time is it?\" "
        "Respond ONLY with JSON in the form {\"tool_name\": \"...\", \"args\": {}}."
    )},
    {"type": "tool", "prompt": (
        "You have a tool called \"calculate\" with args {expression: string}. "
        "The user said: \"What's 47 times 23?\" "
        "Respond ONLY with JSON in the form {\"tool_name\": \"...\", \"args\": {...}}."
    )},
    {"type": "tool", "prompt": (
        "You have a tool called \"open_file\" with args {filename: string}. "
        "The user said: \"Open my notes file.\" "
        "Respond ONLY with JSON in the form {\"tool_name\": \"...\", \"args\": {...}}."
    )},
]


def ram_free_mb():
    if not HAVE_PSUTIL:
        return None
    return psutil.virtual_memory().available / (1024 * 1024)


def run_prompt(model, prompt, use_json_format):
    start = time.time()
    first_token_time = None
    full_text = ""

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "think": False,  # Qwen3.5 auto-enables thinking mode unless explicitly disabled
        "options": {"temperature": 0.3},
    }
    if use_json_format:
        kwargs["format"] = "json"

    thinking_chars = 0
    for chunk in ollama.chat(**kwargs):
        if first_token_time is None:
            first_token_time = time.time() - start
        msg = chunk["message"]
        full_text += msg.get("content", "") or ""
        # Some Ollama/model versions ignore think=False silently — this catches that.
        thinking_chars += len(msg.get("thinking", "") or "")

    total_time = time.time() - start
    return full_text, first_token_time, total_time, thinking_chars


def is_json_parseable(text):
    try:
        json.loads(text.strip())
        return True
    except Exception:
        return False


def main():
    print(f"=== Dragoon Phase 0 Validation: {MODEL} ===\n")
    free_before = ram_free_mb()
    if free_before is not None:
        print(f"Free RAM before test: {free_before:.0f} MB")
    else:
        print("psutil not installed — skipping RAM tracking (pip install psutil --break-system-packages to enable)")
    print()

    results = []
    for i, p in enumerate(TEST_PROMPTS, 1):
        use_json = p["type"] in ("json", "tool")
        print(f"[{i}/10] ({p['type']}) running...", end=" ", flush=True)
        try:
            text, first_tok, total, thinking_chars = run_prompt(MODEL, p["prompt"], use_json)
            entry = {
                "type": p["type"],
                "prompt": p["prompt"],
                "response": text,
                "first_token_s": round(first_tok, 2) if first_tok else None,
                "total_s": round(total, 2),
                "json_valid": is_json_parseable(text) if use_json else None,
                "thinking_chars": thinking_chars,
                "response_chars": len(text),
            }
            results.append(entry)
            think_flag = " [THINK MODE STILL ON]" if thinking_chars > 0 else ""
            print(f"first_token={entry['first_token_s']}s total={entry['total_s']}s "
                  f"json_valid={entry['json_valid']} resp_len={len(text)}{think_flag}")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({"type": p["type"], "prompt": p["prompt"], "error": str(e)})

    ram_after = ram_free_mb()

    print("\n=== SUMMARY ===")
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    if ok:
        first_vals = [r["first_token_s"] for r in ok if r["first_token_s"] is not None]
        avg_first = sum(first_vals) / len(first_vals) if first_vals else None
        avg_total = sum(r["total_s"] for r in ok) / len(ok)
        print(f"Avg first-token latency: {avg_first:.2f}s" if avg_first else "Avg first-token latency: n/a")
        print(f"Avg total latency: {avg_total:.2f}s")
    else:
        avg_total = None
        print("No successful calls — check Ollama is running (`ollama serve`) and the model is pulled.")

    if failed:
        print(f"Failed calls: {len(failed)}/10 — see errors above")

    json_tests = [r for r in ok if r.get("json_valid") is not None]
    json_pass = sum(1 for r in json_tests if r["json_valid"])
    if json_tests:
        print(f"JSON/tool-call parse rate: {json_pass}/{len(json_tests)}")

    if ram_after is not None and free_before is not None:
        print(f"Free RAM after test: {ram_after:.0f} MB (delta: {ram_after - free_before:+.0f} MB)")

    print("\n=== DECISION GATE (Phase 0, build plan) ===")
    if avg_total is not None:
        if avg_total > 18:
            print(f"-> Avg total latency {avg_total:.1f}s exceeds the ~15-20s guideline.")
            print(f"   Action: re-run this script against qwen3.5:2b before proceeding to Phase 1.")
        else:
            print(f"-> Avg total latency {avg_total:.1f}s is within range. Proceed with {MODEL}.")

    if json_tests:
        pass_rate = json_pass / len(json_tests)
        if pass_rate < 0.8:  # equivalent to the build plan's "8/10" — applied as a rate, not a raw count
            print(f"-> JSON parse rate {json_pass}/{len(json_tests)} ({pass_rate:.0%}) is BELOW the 80% exit criterion.")
            print(f"   Action: fix prompt/format handling before starting Phase 1 — do not proceed on a partial pass.")
        else:
            print(f"-> JSON parse rate {json_pass}/{len(json_tests)} ({pass_rate:.0%}) meets the Phase 0 exit criterion.")

    still_thinking = sum(1 for r in ok if r.get("thinking_chars", 0) > 0)
    if still_thinking:
        print(f"-> WARNING: {still_thinking}/{len(ok)} calls still emitted thinking-mode output despite think=False.")
        print(f"   This means the flag isn't being honored — likely an Ollama/model version issue, not your setup.")
        print(f"   Check `ollama --version` is current; this needs investigating before latency numbers mean anything.")

    with open("phase0_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRaw results saved to phase0_results.json")


if __name__ == "__main__":
    main()
