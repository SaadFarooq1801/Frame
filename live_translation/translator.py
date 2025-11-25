import asyncio
from frame_sdk import Frame
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def display_text_scroll(frame, text):
    """
    Displays long text by scrolling down the screen line by line.
    """
    max_width = 120
    lines = []

    # Break long string into multiple display-safe segments
    while len(text) > max_width:
        lines.append(text[:max_width])
        text = text[max_width:]
    lines.append(text)

    # Show each chunk for readability
    for line in lines:
        await frame.display.set_text(line)
        await frame.display.show()
        await asyncio.sleep(1.5)


async def main():
    async with Frame() as f:

        await f.display.set_text("Translator Ready...")
        await f.display.show()

        print("Listening continuously from Frame mic...")

        while True:
            # ---------- RECORD FROM FRAME MIC ----------
            audio = await f.microphone.record_audio(
                silence_cutoff_length_in_seconds=1,
                max_length_in_seconds=5
            )

            if len(audio) == 0:
                continue

            print("Audio captured. Sending to Whisper...")

            # ---------- SEND AUDIO TO WHISPER ----------
            wav_bytes = audio.astype("int16").tobytes()

            whisper_result = await client.audio.transcriptions.create(
                model="gpt-4o-mini-tts",
                file=("speech.wav", wav_bytes),
                response_format="verbose_json"
            )

            text = whisper_result.text.strip()
            if text == "":
                continue

            print("Heard:", text)

            # ---------- TRANSLATE ----------
            result = await client.responses.create(
                model="gpt-4.1-mini",
                input=f"Translate this to English clearly and concisely: {text}"
            )

            translated = result.output_text.strip()
            print("Translated:", translated)

            # ---------- DISPLAY ON FRAME ----------
            await display_text_scroll(f, translated)


if __name__ == "__main__":
    asyncio.run(main())
