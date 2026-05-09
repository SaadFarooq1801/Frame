import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import asyncio
import os
import queue
import threading
import tkinter as tk

import speech_recognition as sr
from elevenlabs import ElevenLabs
from groq import Groq

# --- Config ---
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# --- Clients ---
eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
recognizer = sr.Recognizer()

# --- Glasses simulation (Tkinter) ---
_ui_queue: queue.Queue = queue.Queue()
_tk_root = None
_stop_flag = threading.Event()


def _build_glasses_window():
    global _tk_root
    _tk_root = tk.Tk()
    _tk_root.title("Frame Glasses Simulator")
    _tk_root.configure(bg="#0a0a0a")
    _tk_root.geometry("400x150")
    _tk_root.resizable(False, False)

    border = tk.Frame(_tk_root, bg="#2a2a2a", padx=2, pady=2)
    border.pack(expand=True, fill="both", padx=12, pady=12)

    inner = tk.Frame(border, bg="#0a0a0a")
    inner.pack(expand=True, fill="both")

    label = tk.Label(
        inner,
        text="Starting...",
        bg="#0a0a0a",
        fg="#e8e8e8",
        font=("Courier New", 16, "bold"),
        wraplength=360,
        justify="center",
        padx=10,
        pady=10,
    )
    label.pack(expand=True)

    def poll():
        if _stop_flag.is_set():
            _tk_root.quit()
            return
        try:
            while True:
                text = _ui_queue.get_nowait()
                label.config(text=text)
        except queue.Empty:
            pass
        _tk_root.after(80, poll)

    _tk_root.protocol("WM_DELETE_WINDOW", lambda: (_stop_flag.set(), _tk_root.quit()))
    _tk_root.after(80, poll)
    _tk_root.mainloop()


def glasses_show(text: str):
    _ui_queue.put(text)


# --- Audio recording ---
def _record_blocking(timeout: int = 10, phrase_limit: int = 15) -> sr.AudioData:
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        return recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)


async def record_audio(timeout: int = 10, phrase_limit: int = 15):
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _record_blocking, timeout, phrase_limit)
    except sr.WaitTimeoutError:
        return None
    except Exception as e:
        print(f"Recording error: {e}")
        return None


# --- ElevenLabs STT ---
def _transcribe_blocking(audio: sr.AudioData) -> tuple[str, str]:
    wav_bytes = audio.get_wav_data()
    result = eleven.speech_to_text.convert(
        file=("audio.wav", wav_bytes, "audio/wav"),
        model_id="scribe_v1",
    )
    text = result.text.strip() if result.text else ""
    lang = getattr(result, "language_code", "unknown") or "unknown"
    return text, lang


async def transcribe(audio: sr.AudioData) -> tuple[str, str]:
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _transcribe_blocking, audio)
    except Exception as e:
        print(f"Transcription error: {e}")
        return "", "unknown"


# --- Groq: translate to English only ---
def _translate_blocking(text: str, source_lang: str) -> str:
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a translator. Your only job is to translate text into English. Output the English translation only — no explanations, no original text, no other language.",
            },
            {
                "role": "user",
                "content": f"Translate this to English: {text}",
            },
        ],
        max_tokens=200,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


# --- Groq: simplify English text ---
def _simplify_blocking(text: str) -> str:
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite this in plain, simple English for smart glasses. "
                    "Max 8 words. Return ONLY the rewritten text."
                ),
            },
            {"role": "user", "content": text},
        ],
        max_tokens=50,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


async def translate_text(text: str, source_lang: str) -> str:
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None, _translate_blocking, text, source_lang
        )
        print(f"Translated: '{result}'")
        return result
    except Exception as e:
        print(f"Groq error: {e}")
        return text


async def simplify(text: str) -> str:
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, _simplify_blocking, text)
        print(f"Simplified: '{result}'")
        return result
    except Exception as e:
        print(f"Groq error: {e}")
        return text


# --- Mode selection via voice ---
async def pick_mode() -> str:
    """Ask the user to say 'translate' or 'simplify'. Returns 'translate' or 'simplify'."""
    while not _stop_flag.is_set():
        glasses_show("Say: TRANSLATE\nor SIMPLIFY")
        print("\nSay 'translate' or 'simplify'...")

        audio = await record_audio(timeout=10, phrase_limit=5)
        if audio is None:
            print("No speech — trying again")
            continue

        text, _ = await transcribe(audio)
        text_lower = text.lower()
        print(f"Heard: '{text}'")

        if "translat" in text_lower:
            glasses_show("Mode: TRANSLATE")
            print("Mode selected: TRANSLATE")
            await asyncio.sleep(1)
            return "translate"
        elif "simplif" in text_lower:
            glasses_show("Mode: SIMPLIFY")
            print("Mode selected: SIMPLIFY")
            await asyncio.sleep(1)
            return "simplify"
        else:
            glasses_show("Didn't catch that.\nSay TRANSLATE\nor SIMPLIFY")
            await asyncio.sleep(1.5)


# --- Translation loop ---
async def run_translation():
    print("\n=== TRANSLATION MODE ===")
    glasses_show("TRANSLATE mode\nListening...")

    while not _stop_flag.is_set():
        print("\nListening...")
        glasses_show("Listening...")

        audio = await record_audio()
        if audio is None:
            glasses_show("No speech heard")
            await asyncio.sleep(1)
            continue

        text, lang = await transcribe(audio)
        if not text:
            glasses_show("Could not hear")
            await asyncio.sleep(1)
            continue

        print(f"Heard ({lang}): '{text}'")
        glasses_show("Processing...")

        if lang in ("en", "english", "unknown"):
            display = text
        else:
            display = await translate_text(text, lang)

        glasses_show(display)
        read_time = max(5.0, len(display.split()) * 0.45)
        await asyncio.sleep(read_time)


# --- Simplification loop ---
async def run_simplification():
    print("\n=== SIMPLIFICATION MODE ===")
    glasses_show("SIMPLIFY mode\nListening...")

    while not _stop_flag.is_set():
        print("\nListening...")
        glasses_show("Listening...")

        audio = await record_audio()
        if audio is None:
            glasses_show("No speech heard")
            await asyncio.sleep(1)
            continue

        text, _ = await transcribe(audio)
        if not text:
            glasses_show("Could not hear")
            await asyncio.sleep(1)
            continue

        print(f"Heard: '{text}'")
        glasses_show("Simplifying...")

        display = await simplify(text)
        glasses_show(display)
        read_time = max(5.0, len(display.split()) * 0.45)
        await asyncio.sleep(read_time)


# --- Main ---
async def main():
    print("=" * 60)
    print("FRAME LIVE TRANSLATOR / SIMPLIFIER")
    print("Close the window to stop")
    print("=" * 60)

    mode = await pick_mode()

    if mode == "translate":
        await run_translation()
    else:
        await run_simplification()


# --- Entry point (Tkinter on main thread, asyncio in thread) ---
def run():
    loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main())
        except Exception as e:
            print(f"Error: {e}")
        finally:
            _stop_flag.set()

    t = threading.Thread(target=run_loop, daemon=True)
    t.start()

    try:
        _build_glasses_window()
    except KeyboardInterrupt:
        pass
    finally:
        _stop_flag.set()
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=3)
        print("Stopped.")


if __name__ == "__main__":
    run()
