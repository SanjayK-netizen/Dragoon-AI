"""
Dragoon — tests/test_candidates.py (Phase 2 exit criteria)

Run: python Tests/test_candidates.py

Build plan Phase 2 exit criteria:
  - >=90% correct auto-execution on CLEAR commands
  - ZERO silent wrong-executions on AMBIGUOUS commands (must route to
    disambiguate, never auto_execute)

"Correct" for clear commands = action was auto_execute (the right call for
an unambiguous input). For ambiguous ones, the only acceptable action is
disambiguate — auto_execute on an ambiguous input is a hard failure
regardless of score, because that's the exact failure mode this phase
exists to prevent.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.candidates import generate_and_score  # noqa: E402

# 25 CLEAR commands — close to a known vocabulary entry, should auto_execute.
CLEAR_COMMANDS = [
    "Set a reminder for 5pm",
    "Remind me to call mom at 6",
    "Delete my last reminder",
    "Remove the reminder about the meeting",
    "Open my notes file",
    "Open the budget spreadsheet",
    "Calculate 47 times 23",
    "What's 18% of 250?",
    "What time is it right now",
    "Tell me the current time",
    "Add milk to my shopping list",
    "Put eggs on the grocery list",
    "Send a message to Alex",
    "Text mom that I'll be late",
    "Start a timer for 10 minutes",
    "Set a 5 minute timer",
    "Check the weather forecast",
    "What's the weather like tomorrow",
    "Set a reminder to take medicine at 9am",
    "Open the file called notes.txt",
    "Calculate the square root of 144",
    "Add bread to my list",
    "Start a 20 minute timer",
    "Check today's weather",
    "Send a message to the team",
]

# 15 AMBIGUOUS / edge-case commands — vague, multi-interpretation, or far
# from the known vocabulary. Must route to disambiguate, not guess.
AMBIGUOUS_COMMANDS = [
    "Do that thing again",
    "Handle it",
    "You know what I mean",
    "Set it up",
    "Take care of the thing from before",
    "Fix that",
    "Make it happen",
    "Do the usual",
    "Sort this out",
    "Send it",
    "Cancel that",
    "Undo the last one",
    "Get it done",
    "Put that away",
    "Deal with the reminder thing",
]


def run_candidate_test():
    print(f"=== CLEAR COMMANDS ({len(CLEAR_COMMANDS)}) ===\n")
    clear_correct = 0
    for text in CLEAR_COMMANDS:
        result = generate_and_score(text)
        action = result["action"]
        ok = action == "auto_execute"
        clear_correct += ok
        best = result["candidates"][result["selected_index"]] if result["selected_index"] is not None else None
        score = best["combined_score"] if best else None
        print(f"[{'OK ' if ok else 'FAIL'}] action={action:12s} score={score} text={text!r}")

    print(f"\n=== AMBIGUOUS COMMANDS ({len(AMBIGUOUS_COMMANDS)}) ===\n")
    ambiguous_safe = 0
    silent_wrong = []
    for text in AMBIGUOUS_COMMANDS:
        result = generate_and_score(text)
        action = result["action"]
        ok = action == "disambiguate"
        ambiguous_safe += ok
        best = result["candidates"][result["selected_index"]] if result["selected_index"] is not None else None
        score = best["combined_score"] if best else None
        print(f"[{'OK ' if ok else 'FAIL — SILENT WRONG-EXECUTION'}] action={action:12s} score={score} text={text!r}")
        if not ok:
            silent_wrong.append(text)

    clear_rate = clear_correct / len(CLEAR_COMMANDS)
    ambiguous_rate = ambiguous_safe / len(AMBIGUOUS_COMMANDS)

    print(f"\n=== RESULT ===")
    print(f"Clear commands auto-executed correctly: {clear_correct}/{len(CLEAR_COMMANDS)} ({clear_rate:.1%})")
    print(f"Ambiguous commands safely disambiguated: {ambiguous_safe}/{len(AMBIGUOUS_COMMANDS)} ({ambiguous_rate:.1%})")

    print(f"\n=== EXIT CRITERIA (Phase 2, build plan) ===")
    if clear_rate >= 0.90:
        print(f"-> Clear-command rate {clear_rate:.1%} meets the >=90% threshold.")
    else:
        print(f"-> Clear-command rate {clear_rate:.1%} is BELOW the 90% threshold. "
              f"Tune AUTO_EXECUTE_THRESHOLD or KNOWN_COMMANDS in core/candidates.py.")

    if silent_wrong:
        print(f"-> HARD FAIL: {len(silent_wrong)} ambiguous command(s) were auto-executed instead "
              f"of disambiguated. This is the exact failure mode Phase 2 exists to prevent — "
              f"do not proceed to Phase 3 until this is zero:")
        for t in silent_wrong:
            print(f"     {t!r}")
    else:
        print(f"-> Zero silent wrong-executions on ambiguous commands. Exit criterion met.")


if __name__ == "__main__":
    run_candidate_test()
