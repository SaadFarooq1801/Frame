import asyncio
import speech_recognition as sr
from googletrans import Translator
from frame_msg import FrameMsg

async def main():
    frame = FrameMsg()
    await frame.connect()

    # Upload standard Lua libraries and your translator app
    await frame.upload_stdlua_libs(lib_names=['data'])
    await frame.upload_frame_app(local_filename="lua/translator_frame_app.lua")
    await frame.start_frame_app()
    frame.attach_print_response_handler()

    recognizer = sr.Recognizer()
    translator = Translator()

    await frame.print_short_text("Translator ready!")

    try:
        while True:
            with sr.Microphone() as source:
                print("Listening...")
                audio = recognizer.listen(source)

            try:
                # Convert speech to text
                text = recognizer.recognize_google(audio)
                print(f"Original: {text}")

                # Translate to English
                translated = translator.translate(text, dest='en')
                print(f"Translated: {translated.text}")

                # Send translated text to Frame
                await frame.send_message(0x30, translated.text)

            except Exception as e:
                print(f"Error: {e}")
                await frame.send_message(0x30, "Error processing audio")

    finally:
        await frame.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
