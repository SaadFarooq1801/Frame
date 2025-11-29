import asyncio
from frame_msg import FrameMsg, RxAudio, TxCode
from googletrans import Translator
import speech_recognition as sr
import io
import wave

# Initialize translator
translator = Translator()

# Configuration
TARGET_LANGUAGE = "en"  # en, es, fr, de, ar, zh-cn, ja, etc.
RECORDING_DURATION = 7  # seconds per recording

async def display_text_on_frame(frame, text, duration=3):
    """
    Display text on Frame glasses.
    """
    try:
        await frame.print_short_text(text)
        await asyncio.sleep(duration)
    except Exception as e:
        print(f"Display error: {e}")

def transcribe_audio_from_wav(wav_bytes):
    """
    Transcribe audio from WAV bytes using Google Speech Recognition.
    """
    recognizer = sr.Recognizer()
    
    # Adjust settings for better accuracy
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    
    try:
        # Create audio file from bytes
        audio_file = io.BytesIO(wav_bytes)
        
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            
            print("🔄 Transcribing...")
            text = recognizer.recognize_google(audio, language=None)  # Auto-detect
            return text.strip()
    
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"❌ Speech Recognition API error: {e}")
        return ""
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""

def translate_text(text, target_lang="en"):
    """
    Translate text using Google Translate.
    """
    try:
        result = translator.translate(text, dest=target_lang)
        return result.text
    except Exception as e:
        print(f"Translation error: {e}")
        return None

def detect_language(text):
    """
    Detect the language of text.
    """
    try:
        result = translator.detect(text)
        return result.lang
    except Exception as e:
        print(f"Language detection error: {e}")
        return "unknown"

async def record_audio_from_frame(frame, duration=5):
    """
    Record audio from Frame microphone and return WAV bytes.
    """
    try:
        # Hook up the RxAudio receiver
        rx_audio = RxAudio()
        audio_queue = await rx_audio.attach(frame)
        
        # Start streaming audio
        print(f"🎤 Recording for {duration} seconds...")
        await frame.send_message(0x30, TxCode(value=1).pack())
        
        # Record for specified duration
        await asyncio.sleep(duration)
        
        # Stop streaming
        await frame.send_message(0x30, TxCode(value=0).pack())
        
        # Get the audio samples
        audio_samples = await asyncio.wait_for(audio_queue.get(), timeout=10.0)
        
        # Convert to WAV bytes
        wav_bytes = RxAudio.to_wav_bytes(audio_samples)
        
        # Clean up
        rx_audio.detach(frame)
        
        print(f"✅ Captured {len(wav_bytes)} bytes of audio")
        return wav_bytes
    
    except asyncio.TimeoutError:
        print("❌ Timeout waiting for audio")
        return None
    except Exception as e:
        print(f"❌ Recording error: {e}")
        return None

async def main():
    frame = FrameMsg()
    
    try:
        print("=" * 60)
        print("FRAME LIVE TRANSLATOR")
        print("=" * 60)
        print(f"Target Language: {TARGET_LANGUAGE}")
        print(f"Recording Duration: {RECORDING_DURATION} seconds")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        # Connect to Frame
        print("\n📡 Connecting to Frame...")
        await frame.connect()
        
        await display_text_on_frame(frame, "Connecting...", 1)
        
        # Check battery
        batt_mem = await frame.send_lua(
            'print(frame.battery_level() .. " / " .. collectgarbage("count"))', 
            await_print=True
        )
        print(f"🔋 Battery/Memory: {batt_mem}")
        
        # Upload standard Lua libraries
        print("📤 Uploading Lua libraries...")
        await frame.upload_stdlua_libs(lib_names=['data', 'code', 'audio'])
        
        # Upload the Frame app
        print("📤 Uploading Frame app...")
        await frame.upload_frame_app(local_filename="lua/audio_frame_app.lua")
        
        # Attach print handler
        frame.attach_print_response_handler()
        
        # Start the Frame app
        print("🚀 Starting Frame app...")
        await frame.start_frame_app()
        
        await display_text_on_frame(frame, "Ready!", 2)
        
        print("\n✅ Frame initialized successfully!\n")
        
        # Main translation loop
        recording_count = 0
        
        while True:
            recording_count += 1
            print(f"\n{'='*60}")
            print(f"Recording #{recording_count}")
            print(f"{'='*60}")
            
            # Show listening indicator
            await display_text_on_frame(frame, "Listening...", 0.5)
            
            # Record audio from Frame
            wav_bytes = await record_audio_from_frame(frame, RECORDING_DURATION)
            
            if wav_bytes is None or len(wav_bytes) == 0:
                print("⚠️  No audio captured")
                await display_text_on_frame(frame, "No audio", 1)
                continue
            
            # Show processing status
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
            
            # Skip translation if already in target language
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
            
            # Display translation on Frame
            await display_text_on_frame(frame, translated, 5)
            
            print("\n⏳ Ready for next recording...")
            await asyncio.sleep(1)
    
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
        # Clean disconnection
        try:
            frame.detach_print_response_handler()
            await frame.stop_frame_app()
            await frame.disconnect()
            print("✅ Disconnected from Frame")
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
