import asyncio
import speech_recognition as sr
from googletrans import Translator
from frame_sdk import Frame

# Helper: wrap raw audio bytes into a WAV buffer compatible with SpeechRecognition
def assemble_wav(raw_audio):
    import io
    import wave

    # Frame mic audio is 16-bit PCM, 16kHz, mono
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)       # 16-bit audio
        wf.setframerate(16000)
        wf.writeframes(raw_audio)

    wav_buffer.seek(0)
    return wav_buffer

async def main():
    r = sr.Recognizer()
    translator = Translator()

    async with Frame() as f:
        await f.display_text("Translator Ready", x=1, y=1)

        print("Listening through Frame mic... (Ctrl+C to stop)")

        buffer = bytearray()

        async for chunk in f.microphone_stream():
            if chunk is None:
                continue

            # Add chunk to buffer
            buffer.extend(chunk)

            # When buffer is large enough (~0.6s of audio) → process
            if len(buffer) > 16000 * 1:   # 1 second of audio
                wav_data = assemble_wav(bytes(buffer))

                try:
                    audio = sr.AudioFile(wav_data)
                    with audio as source:
                        audio_data = r.record(source)

                    # STT: convert speech to text
                    text = r.recognize_google(audio_data)
                    print("Heard:", text)

                    # Translate to English
                    translated = translator.translate(text, dest="en").text
                    print("Translated:", translated)

                    # Display on Frame
                    await f.display_text(translated, x=1, y=1)

                except Exception as e:
                    print("Error:", e)

                buffer = bytearray()    # reset buffer

if __name__ == "__main__":
    asyncio.run(main())
