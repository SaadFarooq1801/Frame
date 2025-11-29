import asyncio
from frame_msg import FrameMsg, RxAudio, TxCode
from googletrans import Translator
import speech_recognition as sr
import io

# Initialize translator
translator = Translator()

# Configuration
TARGET_LANGUAGE = "en"
RECORDING_DURATION = 5

async def display_text_on_frame(frame, text, duration=3):
    """Display text on Frame glasses."""
    try:
        await frame.print_short_text(text)
        await asyncio.sleep(duration)
    except Exception as e:
        print(f"Display error: {e}")

def transcribe_audio_from_wav(wav_bytes):
    """Transcribe audio from WAV bytes using Google Speech Recognition."""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    
    try:
        audio_file = io.BytesIO(wav_bytes)
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            print("🔄 Transcribing...")
            text = recognizer.recognize_google(audio, language=None)
            return text.strip()
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"❌ API error: {e}")
        return ""
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""

def translate_text(text, target_lang="en"):
    """Translate text using Google Translate."""
    try:
        result = translator.translate(text, dest=target_lang)
        return result.text
    except Exception as e:
        print(f"Translation error: {e}")
        return None

def detect_language(text):
    """Detect the language of text."""
    try:
        result = translator.detect(text)
        return result.lang
    except Exception as e:
        return "unknown"

async def record_audio_from_frame(frame, rx_audio, duration=5):
    """Record audio from Frame microphone and return WAV bytes."""
    try:
        # Clear any pending audio
        while not rx_audio.audio_queue.empty():
            try:
                rx_audio.audio_queue.get_nowait()
            except:
                break
        
        print(f"🎤 Recording for {duration} seconds - SPEAK NOW!")
        
        # Start streaming
        await frame.send_message(0x30, TxCode(value=1).pack())
        await asyncio.sleep(0.2)  # Give it time to start
        
        # Record for duration
        await asyncio.sleep(duration)
        
        # Stop streaming
        await frame.send_message(0x30, TxCode(value=0).pack())
        await asyncio.sleep(0.2)  # Give it time to flush
        
        # Get audio samples with longer timeout
        print("⏳ Waiting for audio data...")
        audio_samples = await asyncio.wait_for(
            rx_audio.audio_queue.get(), 
            timeout=15.0
        )
        
        if len(audio_samples) == 0:
            print("⚠️  Empty audio buffer")
            return None
        
        # Convert to WAV
        wav_bytes = RxAudio.to_wav_bytes(audio_samples)
        print(f"✅ Captured {len(wav_bytes)} bytes ({len(audio_samples)} samples)")
        
        return wav_bytes
    
    except asyncio.TimeoutError:
        print("❌ Timeout - no audio received from Frame")
        print("   Check: Is Frame mic working? Try tapping the Frame.")
        return None
    except Exception as e:
        print(f"❌ Recording error: {e}")
        import traceback
        traceback.print_exc()
        return None

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
        
        # Upload Frame app
        print("📤 Uploading Frame app...")
        await frame.upload_frame_app(local_filename="lua/audio_frame_app.lua")
        
        # Attach handlers
        frame.attach_print_response_handler()
        
        # Start Frame app
        print("🚀 Starting Frame app...")
        await frame.start_frame_app()
        
        # Set up RxAudio ONCE and keep it attached
        print("🎧 Setting up audio receiver...")
        rx_audio = RxAudio()
        audio_queue = await rx_audio.attach(frame)
        rx_audio.audio_queue = audio_queue
        
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
            wav_bytes = await record_audio_from_frame(frame, rx_audio, RECORDING_DURATION)
            
            if wav_bytes is None or len(wav_bytes) == 0:
                print("⚠️  No audio captured")
                await display_text_on_frame(frame, "No audio", 1)
                await asyncio.sleep(1)
                continue
            
            # Process
            await display_text_on_frame(frame, "Processing...", 0.5)
            
            # Transcribe
            text = transcribe_audio_from_wav(wav_bytes)
            
            if text == "":
                print("❌ No speech recognized")
                await display_text_on_frame(frame, "No speech", 2)
                continue
            
            # Detect language
            detected_lang = detect_language(text)
            print(f"✅ Heard ({detected_lang}): '{text}'")
            
            # Skip if already in target language
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
