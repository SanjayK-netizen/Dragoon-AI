"""
Dragoon — tests/test_intent_accuracy.py (Phase 1 exit criteria)

Run: python tests/test_intent_accuracy.py

Build plan Phase 1 calls for 60-100 labeled utterances, 20-30 per category.
This starter set has 30 (10 per category) so the harness is runnable today —
expand LABELED_SET below before trusting the result; 30 examples is not
enough to reliably claim 85%, it's enough to confirm the script works.

Exit criteria: >=85% accuracy, reproduced on two separate runs.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.intent import classify_intent  # noqa: E402

# (text, expected_intent) — expand this to 60-100 total before trusting the
# accuracy number. Include deliberately ambiguous ones, per the build plan.
LABELED_SET = [
    # command
    ("Set a reminder for 5pm", "command"),
    ("Remind me to call mom tomorrow", "command"),
    ("Open my notes file", "command"),
    ("Calculate 47 times 23", "command"),
    ("Turn off the lights", "command"),
    ("Add milk to my shopping list", "command"),
    ("Delete the last reminder", "command"),
    ("Play some music", "command"),
    ("Set a timer for ten minutes", "command"),
    ("Save this to my notes", "command"),
    # question
    ("What time is it?", "question"),
    ("What's the weather like today?", "question"),
    ("How many days until Friday?", "question"),
    ("What's on my calendar tomorrow?", "question"),
    ("What is the capital of France?", "question"),
    ("How do I convert celsius to fahrenheit?", "question"),
    ("What reminders do I have set?", "question"),
    ("Who won the game last night?", "question"),
    ("What's 15% of 200?", "question"),
    ("Is it going to rain today?", "question"),
    # conversation
    ("Hey, how's it going?", "conversation"),
    ("Good morning", "conversation"),
    ("Tell me a joke", "conversation"),
    ("That's really helpful, thanks", "conversation"),
    ("I'm bored", "conversation"),
    ("You're pretty good at this", "conversation"),
    ("Never mind", "conversation"),
    ("Goodnight", "conversation"),
    ("Just testing you out", "conversation"),
    ("What's up?", "conversation"),
]


def run_accuracy_test():
    correct = 0
    mismatches = []

    print(f"Running {len(LABELED_SET)} labeled utterances through classify_intent()...\n")

    for text, expected in LABELED_SET:
        result = classify_intent(text)
        actual = result["intent"]
        ok = actual == expected
        correct += ok
        status = "OK " if ok else "FAIL"
        print(f"[{status}] expected={expected:12s} got={actual:12s} text={text!r}")
        if not ok:
            mismatches.append((text, expected, actual))

    accuracy = correct / len(LABELED_SET)

    print(f"\n=== RESULT ===")
    print(f"Accuracy: {correct}/{len(LABELED_SET)} ({accuracy:.1%})")

    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        for text, expected, actual in mismatches:
            print(f"  expected={expected} got={actual}  {text!r}")

    print(f"\n=== EXIT CRITERIA (Phase 1, build plan) ===")
    if accuracy >= 0.85:
        print(f"-> {accuracy:.1%} meets the 85% threshold. Run this once more — "
              f"exit criteria requires reproducing this on a second independent run, "
              f"not just one pass.")
    else:
        print(f"-> {accuracy:.1%} is BELOW the 85% threshold. Do not proceed to Phase 2. "
              f"Review the mismatches above, iterate on the prompt/few-shot examples "
              f"in core/intent.py, and re-run.")

    if len(LABELED_SET) < 60:
        print(f"\nNOTE: only {len(LABELED_SET)} examples in this set — build plan calls for "
              f"60-100. This result is a smoke test, not a trustworthy accuracy measurement "
              f"yet. Expand LABELED_SET before treating {accuracy:.1%} as real.")


if __name__ == "__main__":
    run_accuracy_test()
