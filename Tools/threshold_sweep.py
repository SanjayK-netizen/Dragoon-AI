import json
import core.candidates as c
from Tests.test_candidates import CLEAR_COMMANDS, AMBIGUOUS_COMMANDS


def evaluate():
    results = []
    auto_range = [round(x,2) for x in [0.6 + i*0.05 for i in range(8)]]  # 0.6..0.95
    agree_range = [round(x,2) for x in [0.6 + i*0.05 for i in range(8)]]
    for auto in auto_range:
        for agree in agree_range:
            c.AUTO_EXECUTE_THRESHOLD = auto
            c.AGREEMENT_THRESHOLD = agree
            clear_correct = 0
            ambiguous_safe = 0
            silent_wrong = []
            for text in CLEAR_COMMANDS:
                res = c.generate_and_score(text)
                if res["action"] == "auto_execute":
                    clear_correct += 1
            for text in AMBIGUOUS_COMMANDS:
                res = c.generate_and_score(text)
                if res["action"] == "disambiguate":
                    ambiguous_safe += 1
                else:
                    silent_wrong.append(text)
            clear_rate = clear_correct / len(CLEAR_COMMANDS)
            ambiguous_rate = ambiguous_safe / len(AMBIGUOUS_COMMANDS)
            results.append({
                "AUTO_EXECUTE_THRESHOLD": auto,
                "AGREEMENT_THRESHOLD": agree,
                "clear_correct": clear_correct,
                "clear_rate": clear_rate,
                "ambiguous_safe": ambiguous_safe,
                "ambiguous_rate": ambiguous_rate,
                "silent_wrong_count": len(silent_wrong),
                "silent_wrong_examples": silent_wrong,
            })
            print(f"auto={auto:.2f} agree={agree:.2f} clear={clear_correct}/25 ({clear_rate:.2%}) ambiguous_safe={ambiguous_safe}/15 ({ambiguous_rate:.2%}) silent_wrong={len(silent_wrong)}")
    # find safe configs (zero silent wrong) and sort by clear_rate desc
    safe = [r for r in results if r["silent_wrong_count"] == 0]
    safe_sorted = sorted(safe, key=lambda r: r["clear_rate"], reverse=True)
    out = {
        "total_configs": len(results),
        "safe_count": len(safe_sorted),
        "top_safe": safe_sorted[:10],
    }
    with open("Tools/threshold_sweep_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== Summary ===")
    print(f"Total configs: {len(results)}  Safe configs (0 silent wrongs): {len(safe_sorted)}")
    if safe_sorted:
        best = safe_sorted[0]
        print("Best safe config:")
        print(json.dumps(best, indent=2))
    else:
        print("No safe configs found in the grid.")


if __name__ == '__main__':
    evaluate()
