# Dragoon AI
Local, offline-first voice assistant (qwen3.5:2b via Ollama) with intent classification, memory, and a safety-tiered agent tool loop. CPU-only, solo build, phased 0–10 development.

## Voice input

Install the microphone dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the assistant in microphone mode:

```powershell
python Tests/main.py
```

For headless testing without a microphone, pass one command directly:

```powershell
python Tests/main.py --text "Calculate 12 + 7" --no-tts
```
