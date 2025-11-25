import asyncio
from frame_msg import FrameMsg, RxAudio, TxCode
from openai import OpenAI

client = OpenAI()

async def main():
    frame = FrameMsg()

    try:
        await frame.connect()
        print("Connected to Frame.")

        # Display ready message
        await frame.print_short_text("Translator Ready")

        # Upload essential Lua libs
        await frame.upload_stdlua_libs(["data", "audio", "code"])

        # Upload our custom firmware app
        await frame.upload_frame_app("lua/translator_frame_app.lua")

        # Start app
        frame.attach_print_response_handler()
        await frame.start_frame_app()

        # Set up microphone receiver
        rx = RxAudio(streaming=True)
        audio_queue = await rx.attach(frame)

        # Start microphone
        print("Starting mic...")
        await frame.send_message(0x30, TxCode(value=1).pack())

        print("Listening through Frame mic... (Ctrl+C to stop)")

        while True:
            try:
                samples = audio_queue.get_nowait()

            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)
                continue

            if samples is None:
                continue

            # Convert raw audio to WAV bytes
            wav_bytes = RxAudio.to_wav_bytes(samples)

            # Send to Whisper
            whisper = client.audio.transcriptions.create(
                model="gpt-4o-mini-tts",
                file=wav_bytes,
                response_format="verbose_json"
            )

            spoken_text = whisper.text.strip()
            if not spoken_text:
                continue

            print("Heard:", spoken_text)

            # Translate to English
            translated = client.responses.create(
                model="gpt-4.1-mini",
                input=f"Translate to English: {spoken_text}"
            )

            output = translated.output_text.strip()
            print("Translated:", output)

            # Send to Frame display
            await frame.send_message(0x20, output.encode("utf-8"))

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        # Stop mic
        await frame.send_message(0x30, TxCode(value=0).pack())
        await frame.stop_frame_app()
        await frame.disconnect()
        print("Disconnected.")

if __name__ == "__main__":
    asyncio.run(main())
