import os
import subprocess

import ollama
import pyttsx3
import speech_recognition as sr


WAKE_WORD = "dragoon"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")

recognizer = sr.Recognizer()
speaker = pyttsx3.init()


def speak(text):
    print(f"Dragoon: {text}")
    speaker.say(text)
    speaker.runAndWait()


def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

    try:
        return recognizer.recognize_google(audio).lower().strip()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        speak("Speech recognition requires an internet connection.")
        return ""


def ask_ollama(command):
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": command,
                }
            ],
        )
        return response["message"]["content"]
    except Exception as error:
        print(f"Ollama error: {error}")
        return "I could not connect to Ollama. Check that Ollama is running."


def run_command(command):
    if "open notepad" in command:
        subprocess.Popen("notepad.exe")
        speak("Opening Notepad.")

    elif "open calculator" in command:
        subprocess.Popen("calc.exe")
        speak("Opening Calculator.")

    elif "open browser" in command:
        subprocess.Popen("start msedge", shell=True)
        speak("Opening the browser.")

    elif command in {"stop", "exit", "quit", "goodbye"}:
        speak("Goodbye.")
        return False

    else:
        answer = ask_ollama(command)
        speak(answer)

    return True


def main():
    speak("Voice control is ready. Say Dragoon followed by a command.")

    while True:
        try:
            phrase = listen()

            if WAKE_WORD not in phrase:
                continue

            command = phrase.replace(WAKE_WORD, "").strip()

            if not command:
                speak("Yes?")
                command = listen()

            if not run_command(command):
                break

        except sr.WaitTimeoutError:
            continue
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()