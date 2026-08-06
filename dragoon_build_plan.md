# Dragoon — Build Plan

**Stack:** Python · Ollama (`qwen3.5:4b` + `qwen3-embedding:0.6b`) · SQLite · Whisper · pyttsx3/SAPI5 · CPU-only (AMD Ryzen)

**Confidence key used below:** `[high]` = verified against current sources/docs · `[inference]` = reasoned estimate, not measured · `[assumption]` = stated explicitly because I don't have the data to know it

---

## Definition of Done (applies to every phase below)

> A phase is complete when its exit criteria are met **and reproduced on a second, independent test run** — not on the first pass. Sampling-based models look "done" more often than they are; re-verification is not optional, it's part of the definition.

---

## 0. Reality Check

- "High precision probability of outcomes" is not something a CPU-only 3-4B model does. `[high]` What you're actually building in Phase 2 is **ranked candidates from scored heuristics** (embedding similarity + keyword overlap), not calibrated statistics. Call it that internally too — naming it "probability" will make you misjudge when it's working correctly.
- Timeline below is a planning estimate, not a delivery date. `[inference]` Variance comes mainly from Phase 2 and Phase 4 — the scoring tuning and the agent retry logic are where solo projects like this usually run long.
- RAM: `[assumption — unknown spec]` confirm you have at least 8GB free before Phase 0. `qwen3.5:4b` is 3.4GB on disk; with context and OS overhead, 8GB free is workable, 16GB is comfortable. Check with `wmic OS get FreePhysicalMemory` (Windows) before you start.

---

## 1. Model Choice

| Role | Model | Size | Notes |
|---|---|---|---|
| Primary LLM | `qwen3.5:4b` | 3.4GB | 256K context, native tool-calling, thinking mode available |
| Embedding/scoring | `qwen3-embedding:0.6b` | ~1GB | Used only in Phase 2 candidate scoring |
| Fallback (too slow) | `qwen3.5:2b` | 2.7GB | Drop to this if Phase 0 latency test fails |
| Stretch (if latency budget allows) | `qwen3.5:9b` | 6.6GB | Optional upgrade after Phase 5, not before |

```bash
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
```

---

## 2. Architecture

```
[Mic] → [Whisper STT] → [Intent Parser: qwen3.5:4b, JSON mode]
                                  |
                 ----------------------------------
                 |                                |
          intent=command              intent=question/conversation
                 |                                |
 [Multi-Candidate Gen: N=3, qwen3.5:4b]   [Direct response, memory-informed]
                 |
 [Score: qwen3-embedding:0.6b similarity + keyword overlap]
                 |
        score ≥ threshold? --No--> [Ask user to disambiguate]
                 |Yes
 [Agent Loop: PARSE → SELECT_TOOL → EXECUTE → VERIFY → RESPOND]
                 |
          [SQLite: read/write context]
                 |
          [pyttsx3/SAPI5 TTS] → [Speaker]
```

---

## 3. Environment Setup

```bash
pip install ollama sounddevice numpy openai-whisper pyttsx3 --break-system-packages
# Windows: if pyttsx3's SAPI5 driver errors on import, add:
pip install pywin32 --break-system-packages
```

Suggested repo layout:
```
dragoon/
  core/
    intent.py        # Phase 1
    candidates.py     # Phase 2
    memory.py         # Phase 3
    agent_loop.py      # Phase 4
  tools/
    registry.py
    reminder.py
  io/
    stt.py
    tts.py
  data/
    dragoon.db
  tests/
  main.py
```

---

## 4. Phased Build

### Phase 0 — Environment & Model Validation
**Objective:** Confirm the model runs acceptably on your hardware before building anything on top of it.

1. Confirm Ollama is current: `ollama --version`, update if needed.
2. Pull both models (above).
3. Write a test script sending 10 varied prompts: plain Q&A, a JSON-extraction prompt, a tool-call-style prompt.
4. Log latency with `time.time()` deltas — **measure first-token time and total time separately**, they diagnose different problems.
5. Record tokens/sec and RAM usage during inference (`psutil` or Task Manager).
6. Confirm tool-call/JSON output actually parses as valid JSON on ≥8/10 prompts — this is the step most guides skip, and where the template-mismatch risk above shows up first.
7. **Decision gate:** if average response time exceeds ~15-20s for a short prompt, drop to `qwen3.5:2b` and re-run steps 3-6 before proceeding.

**Exit criteria:** documented latency numbers + ≥8/10 prompts producing parseable structured output, reproduced on a second run.
**Time:** `[inference]` 3-5 days.

---

### Phase 1 — Intent Parser
**Objective:** Single-call classifier: command / question / conversation, structured output only.

1. Define output schema: `{"intent": "command|question|conversation", "raw_text": "string"}`.
2. Use Ollama's `format: "json"` parameter — don't rely on prompt instructions alone to enforce JSON.
3. Write 60-100 labeled test utterances yourself (20-30 per category), including deliberately ambiguous ones.
4. Run the classifier against the set, compute accuracy.
5. Iterate prompt/few-shot examples until accuracy is stable across **two separate runs** (sampling variance means one good run doesn't mean it's fixed).

**Exit criteria:** ≥85% accuracy, reproduced on a second independent run.
**Time:** `[inference]` 1.5-2 weeks.

---

### Phase 2 — Multi-Candidate Generation + Scoring
**Objective:** For command-type input, generate multiple interpretations and rank them. This is the layer you called "probability" — be precise with yourself that it's a scored ranking, not that.

1. Sample N=3 completions at temperature ~0.7-0.9 for classified commands (vary interpretation, not just wording).
2. Embed each candidate + your known command vocabulary with `qwen3-embedding:0.6b`; compute cosine similarity (`numpy`).
3. Score = `0.6 × embedding similarity to nearest known command + 0.4 × keyword overlap`.
4. Threshold (start at 0.75, tune empirically): above → auto-execute top candidate; below → voice-prompt disambiguation ("Did you mean X or Y?").
5. Log every low-confidence case to a file — this becomes your tuning dataset, don't discard it.

**Exit criteria:** on a 40-command mixed test set, ≥90% correct auto-execution on clear commands, **zero silent wrong-executions** on ambiguous ones (they must route to disambiguation, not get guessed wrong).
**Time:** `[inference]` 3-4 weeks — budget extra here; this is where template/parsing issues most often eat time.

---

### Phase 3 — Structured Memory
**Objective:** Queryable state instead of flat chat history.

```sql
CREATE TABLE commands (id INTEGER PRIMARY KEY, ts TEXT, raw_text TEXT, intent TEXT, outcome TEXT);
CREATE TABLE context (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
CREATE TABLE preferences (key TEXT PRIMARY KEY, value TEXT);
```

1. Write `get_context(n=5)` — last N commands + context rows updated in the last 24h.
2. Write `update_context(key, value)` — upsert with timestamp.
3. Inject retrieved context as a bounded block (cap ~500 tokens) into the prompt — don't dump full history.
4. Test staleness: manually insert an old context row, confirm your retrieval window correctly excludes/deprioritizes it.

**Exit criteria:** memory-informed responses correctly reference a fact stated 5+ turns earlier, in ≥8/10 manual test conversations.
**Time:** `[inference]` 2-3 weeks.

---

### Phase 4 — Agent / Tool Loop
**Objective:** A state machine that takes actions, not one that talks about taking them.

1. Define states explicitly as an enum: `PARSE → SELECT_TOOL → EXECUTE → VERIFY → RESPOND`.
2. Tool registry as a plain dict: `{"tool_name": callable}`. Start with 3-4 low-risk tools (reminder written to SQLite, read system time, open a local file, simple calculation). **Don't add destructive tools (file deletion, system commands) until the loop is proven stable.**
3. `SELECT_TOOL`: model outputs tool name + args as JSON, validated against a manual per-tool schema.
4. `EXECUTE`: call the function, catch exceptions explicitly — don't let one bad call crash the loop.
5. `VERIFY`: a cheap deterministic check (did the expected file/DB row appear) — not a second LLM call.
6. On `VERIFY` failure: retry once with the error fed back to the model; on second failure, stop and ask the user directly. No silent third attempt, no infinite loop.

**Exit criteria:** 20 consecutive tool-invoking commands, zero silent failures, zero loops past the retry cap.
**Time:** `[inference]` 4-6 weeks — largest phase; retry/fallback logic is where agent projects most often quietly break.

---

### Phase 5 — Integration + Stress Test
**Objective:** Full pipeline under real conditions.

1. Wire STT → intent parser → candidate scoring → memory-informed prompt → agent loop → TTS end to end.
2. Run a 20-30 turn continuous session: clear commands, ambiguous commands, questions, casual conversation, and at least 3 deliberately adversarial/nonsense inputs.
3. Log per-stage latency separately, not just total — you need to know where time actually goes.
4. Fix state-machine edge cases and re-tune thresholds based on what breaks.

**Exit criteria:** full session completes with no crash, no silent wrong-execution, and per-stage latency logs you can actually read and act on.
**Time:** `[inference]` 2-3 weeks.

---

## 5. Timeline Summary

| Phase | Part-time (8-10 hrs/wk) | Full-time-equivalent (25-30 hrs/wk) |
|---|---|---|
| 0 — Validation | 3-5 days | 2-3 days |
| 1 — Intent Parser | 1.5-2 wks | 4-5 days |
| 2 — Candidates + Scoring | 3-4 wks | 1-1.5 wks |
| 3 — Structured Memory | 2-3 wks | 1 wk |
| 4 — Agent Loop | 4-6 wks | 1.5-2 wks |
| 5 — Integration + Stress Test | 2-3 wks | 1 wk |
| **Total** | **~13.5-19 wks (≈3-4.5 months)** | **~5.5-7.5 wks (≈1.5-2 months)** |

`[assumption]` The part-time column is the one to plan around as a full-time student. The full-time-equivalent column exists for reference, not expectation.

---

## 6. First Checkpoint (this week)

Before writing any pipeline code: finish Phase 0, all 7 steps, both exit criteria met on two separate runs. If step 6 (JSON/tool-call parsing) fails on more than 2/10 prompts, that's your real starting problem — fix the parsing before touching Phase 1.
