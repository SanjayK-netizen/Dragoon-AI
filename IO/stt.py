"""Microphone speech-to-text adapter for Windows."""

import speech_recognition as sr


def transcribe_audio(timeout: float = 5, phrase_time_limit: float = 12) -> str:
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit,
            )
        except sr.WaitTimeoutError:
            return ""

    try:
        text = recognizer.recognize_google(audio)
        print(f"You: {text}")
        return text
    except (sr.UnknownValueError, sr.RequestError, OSError):
        return ""
