"""Windows-native text-to-speech adapter."""

import subprocess


def speak(text: str) -> None:
	if not text:
		return

	escaped = text.replace("'", "''")
	command = (
		"Add-Type -AssemblyName System.Speech; "
		"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
		f"$s.Speak('{escaped}')"
	)
	subprocess.run(
		["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
		check=False,
		creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
	)
