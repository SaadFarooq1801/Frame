import os
import ssl
import time
ssl._create_default_https_context = ssl._create_unverified_context
import asyncio
from frame_msg import FrameMsg, RxAudio, TxCode
from googletrans import Translator
import speech_recognition as sr
import io


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(SCRIPT_DIR, "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

translator = Translator()
TARGET_LANGUAGE = "en"
RECORDING_DURATION = 5

async def display_text_on_frame(frame, text, duration=3):
    try:
        await frame.print_short_text(text)
        await asyncio.sleep(duration)
    except Exception as e:
        print(f"Display error: {e}")

_recognizer = sr.Recognizer()

def _run_whisper(wav_bytes):
    audio_file = io.BytesIO(wav_bytes)
    with sr.AudioFile(audio_file) as source:
        audio = _recognizer.record(source)
    return _recognizer.recognize_whisper(audio, model="small")


async def transcribe_audio_from_wav(wav_bytes):
    loop = asyncio.get_event_loop()
    try:
        print("🔄 Transcribing (local Whisper)...")
        text = await asyncio.wait_for(
            loop.run_in_executor(None, _run_whisper, wav_bytes),
            timeout=60.0  # generous: first run loads model into memory
        )
        return text.strip()
    except asyncio.TimeoutError:
        print("⏱️  Transcription timed out")
        return ""
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        return ""
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""

def translate_text(text, target_lang="en"):
    try:
        result = translator.translate(text, dest=target_lang)
        return result.text
    except Exception as e:
        print(f"Translation error: {e}")
        return None

def detect_language(text):
    try:
        result = translator.detect(text)
        return result.lang
    except Exception as e:
        return "unknown"

async def record_audio_from_frame(frame, rx_audio, duration=5):
    queue = rx_audio.queue

    while not queue.empty():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    print(f"🎤 Recording for {duration} seconds - SPEAK NOW!")
    print("🔍 Verifying Frame app state...")
    try:
        verif = await frame.send_lua("ping()", await_print=True)
        print(f"📡 App Status: {verif}")
        if "TOGGLE_FUNC=true" not in verif:
            print("❌ start_record not found on Frame. Re-uploading app...")
            return "RETRY_UPLOAD"
    except Exception as e:
        print(f"⚠️ Verification error: {e}")

    print("▶️ Starting recording on Frame...")
    await frame.send_lua(f"start_record({duration})")

    timeout = duration + 10
    pcm_chunks = []
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        remaining = deadline - asyncio.get_event_loop().time()
        try:
            chunk = await asyncio.wait_for(queue.get(), timeout=min(remaining, 1.0))
            if chunk is None:
                # None sentinel means the final 0x06 chunk was received — done
                break
            pcm_chunks.append(chunk)
            print(f"  (Received {len(pcm_chunks)} chunks...)", end="\r")
        except asyncio.TimeoutError:
            pass

    print()  # newline after \r progress

    if not pcm_chunks:
        print("⚠️  Empty audio buffer — no chunks received from Frame mic")
        return None

    pcm_data = b"".join(pcm_chunks)
    wav_bytes = RxAudio.to_wav_bytes(pcm_data)
    
    filename = f"capture_{int(time.time())}.wav"
    filepath = os.path.join(RECORDINGS_DIR, filename)
    try:
        with open(filepath, "wb") as f:
            f.write(wav_bytes)
        print(f"💾 Saved to {filepath}")
    except Exception as e:
        print(f"⚠️ Failed to save audio file: {e}")

    print(f"✅ Captured {len(wav_bytes)} bytes ({len(pcm_data)} PCM bytes, {len(pcm_chunks)} chunks)")
    return wav_bytes

async def main():
    frame = FrameMsg()
    rx_audio = None
    
    try:
        print("=" * 60)
        print("FRAME LIVE TRANSLATOR")
        print("=" * 60)
        print(f"Target Language: {TARGET_LANGUAGE}")
        print(f"Recording Duration: {RECORDING_DURATION} seconds")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        # Connect
        print("\n📡 Connecting to Frame...")
        await frame.connect()
        await display_text_on_frame(frame, "Connecting...", 1)
        
        # Check battery
        batt_mem = await frame.send_lua(
            'print(frame.battery_level() .. " / " .. collectgarbage("count"))', 
            await_print=True
        )
        print(f"🔋 Battery/Memory: {batt_mem}")
        
        # Upload Lua libraries
        print("📤 Uploading Lua libraries...")
        await frame.upload_stdlua_libs(lib_names=['data', 'code', 'audio'])
        
        # Upload Frame app - use absolute path
        print("📤 Uploading Frame app...")
        app_path = os.path.join(SCRIPT_DIR, "lua/audio_frame_app.lua")
        await frame.upload_frame_app(local_filename=app_path)
        
        # Attach handlers
        frame.attach_print_response_handler()
        
        # Start Frame app
        print("🚀 Starting Frame app...")
        await frame.start_frame_app()
        
        # Set up RxAudio ONCE and keep it attached
        # streaming=True: chunks arrive immediately on the queue — no waiting for a final sentinel
        print("🎧 Setting up audio receiver...")
        rx_audio = RxAudio(streaming=True)
        await rx_audio.attach(frame)
        
        await display_text_on_frame(frame, "Ready!", 2)
        print("\n✅ Frame initialized successfully!\n")
        
        # Main translation loop
        recording_count = 0
        
        while True:
            recording_count += 1
            print(f"\n{'='*60}")
            print(f"Recording #{recording_count}")
            print(f"{'='*60}")
            
            await display_text_on_frame(frame, "Listening...", 0.5)
            
            # Record audio
            result = await record_audio_from_frame(frame, rx_audio, RECORDING_DURATION)
            
            if result == "RETRY_UPLOAD":
                print("♻️ Re-uploading app...")
                app_path = os.path.join(SCRIPT_DIR, "lua/audio_frame_app.lua")
                await frame.upload_frame_app(local_filename=app_path)
                await frame.start_frame_app()
                await asyncio.sleep(1)
                continue

            wav_bytes = result
            
            if wav_bytes is None or len(wav_bytes) == 0:
                print("⚠️  No audio captured")
                await display_text_on_frame(frame, "No audio", 1)
                await asyncio.sleep(1)
                continue
            
            # Process
            await display_text_on_frame(frame, "Processing...", 0.5)
            
            # Transcribe
            text = await transcribe_audio_from_wav(wav_bytes)
            
            if text == "":
                print("❌ No speech recognized")
                await display_text_on_frame(frame, "No speech", 2)
                continue
            
            # Detect language
            detected_lang = detect_language(text)
            print(f"✅ Heard ({detected_lang}): '{text}'")
            
            # Skips if already in target language
            if detected_lang == TARGET_LANGUAGE:
                print(f"✓ Already in {TARGET_LANGUAGE}")
                await display_text_on_frame(frame, text, 4)
                continue
            
            # Translate
            await display_text_on_frame(frame, "Translating...", 0.5)
            translated = translate_text(text, TARGET_LANGUAGE)
            
            if translated is None or translated == "":
                print("❌ Translation failed")
                await display_text_on_frame(frame, "Failed", 2)
                continue
            
            print(f"🌍 Translated: '{translated}'")
            
            # Display
            await display_text_on_frame(frame, translated, 5)
            
            print("\n⏳ Ready for next recording in 2 seconds...")
            await asyncio.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\n👋 Stopping translator...")
        await display_text_on_frame(frame, "Goodbye!", 1)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await display_text_on_frame(frame, "Error!", 2)
        except:
            pass
    
    finally:
        # Cleanup
        try:
            if rx_audio:
                rx_audio.detach(frame)
            frame.detach_print_response_handler()
            await frame.stop_frame_app()
            await frame.disconnect()
            print("✅ Disconnected from Frame")
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
