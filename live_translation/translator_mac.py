import asyncio
from frame_sdk import Frame
from googletrans import Translator
import speech_recognition as sr

# Initialize
translator = Translator()
recognizer = sr.Recognizer()

# Configuration
TARGET_LANGUAGE = "en"

async def display_text_scroll(frame, text, scroll_delay=2.0):
    """
    Displays text on Frame with word wrapping.
    """
    max_width = 640
    lines = []
    
    words = text.split()
    current_line = ""
    
    for word in words:
        test_line = current_line + word + " "
        if len(test_line) <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line.strip())
            current_line = word + " "
    
    if current_line:
        lines.append(current_line.strip())
    
    for i, line in enumerate(lines):
        await frame.display.show_text(line, 50, 100)
        if i < len(lines) - 1:
            await asyncio.sleep(scroll_delay)
        else:
            await asyncio.sleep(scroll_delay * 1.5)

def record_and_transcribe_from_mac():
    """
    Record audio from Mac microphone and transcribe.
    """
    try:
        with sr.Microphone() as source:
            print("🎤 Listening from Mac microphone...")
            print("   (Adjusting for ambient noise...)")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            print("   Speak now!")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            
            print("🔄 Transcribing...")
            text = recognizer.recognize_google(audio)
            return text.strip()
    
    except sr.WaitTimeoutError:
        print("⏱️  Timeout - no speech detected")
        return ""
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"❌ API error: {e}")
        return ""
    except Exception as e:
        print(f"❌ Error: {e}")
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
    Detect language of text.
    """
    try:
        result = translator.detect(text)
        return result.lang
    except Exception as e:
        print(f"Language detection error: {e}")
        return "unknown"

async def main():
    # Check if Frame is available
    frame_available = True
    frame = None
    
    try:
        # Use async context manager for Frame connection
        async with Frame() as frame:
            print("✅ Frame connected!")
            
            print("=" * 60)
            print("LIVE TRANSLATOR (Mac Microphone → Frame Display)")
            print("=" * 60)
            print(f"Target Language: {TARGET_LANGUAGE}")
            print("Using Mac microphone for input")
            print("Press Ctrl+C to stop")
            print("=" * 60)
            
            await frame.display.show_text("Translator Ready", 50, 100)
            
            try:
                recording_count = 0
                
                while True:
                    recording_count += 1
                    print(f"\n{'='*60}")
                    print(f"Recording #{recording_count}")
                    print(f"{'='*60}")
                    
                    await frame.display.show_text("Listening...", 50, 100)
                    
                    # Record and transcribe from Mac mic
                    text = record_and_transcribe_from_mac()
                    
                    if text == "":
                        await frame.display.show_text("No speech", 50, 100)
                        await asyncio.sleep(1)
                        continue
                    
                    # Detect language
                    detected_lang = detect_language(text)
                    print(f"✅ Heard ({detected_lang}): '{text}'")
                    
                    # Skip translation if already in target language
                    if detected_lang == TARGET_LANGUAGE:
                        print(f"✓ Already in {TARGET_LANGUAGE}")
                        await display_text_scroll(frame, text)
                        continue
                    
                    # Translate
                    await frame.display.show_text("Translating...", 50, 100)
                    
                    translated = translate_text(text, TARGET_LANGUAGE)
                    
                    if translated is None or translated == "":
                        print("❌ Translation failed")
                        await frame.display.show_text("Translation failed", 50, 100)
                        await asyncio.sleep(1)
                        continue
                    
                    print(f"🌍 Translated: '{translated}'")
                    
                    # Display on Frame
                    await display_text_scroll(frame, translated)
                    
                    print("\nReady for next recording...")
                    await asyncio.sleep(1)
            
            except KeyboardInterrupt:
                print("\n\n👋 Stopping translator...")
                await frame.display.show_text("Goodbye!", 50, 100)
                await asyncio.sleep(1)
            
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                await frame.display.show_text("Error occurred", 50, 100)
                await asyncio.sleep(2)
                
    except Exception as e:
        print(f"⚠️  Frame not connected: {e}")
        print("   Running in Mac-only mode (no Frame display)\n")
        frame_available = False
        
        # Fallback: Run without Frame
        print("=" * 60)
        print("LIVE TRANSLATOR (Mac Microphone Only)")
        print("=" * 60)
        print(f"Target Language: {TARGET_LANGUAGE}")
        print("Using Mac microphone for input")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        try:
            recording_count = 0
            
            while True:
                recording_count += 1
                print(f"\n{'='*60}")
                print(f"Recording #{recording_count}")
                print(f"{'='*60}")
                
                # Record and transcribe from Mac mic
                text = record_and_transcribe_from_mac()
                
                if text == "":
                    continue
                
                # Detect language
                detected_lang = detect_language(text)
                print(f"✅ Heard ({detected_lang}): '{text}'")
                
                # Skip translation if already in target language
                if detected_lang == TARGET_LANGUAGE:
                    print(f"✓ Already in {TARGET_LANGUAGE}")
                    print(f"📱 Display: {text}")
                    continue
                
                # Translate
                translated = translate_text(text, TARGET_LANGUAGE)
                
                if translated is None or translated == "":
                    print("❌ Translation failed")
                    continue
                
                print(f"🌍 Translated: '{translated}'")
                print(f"📱 Display: {translated}")
                
                print("\nReady for next recording...")
                await asyncio.sleep(1)
        
        except KeyboardInterrupt:
            print("\n\n👋 Stopping translator...")
        
        except Exception as inner_e:
            print(f"\n❌ Error: {inner_e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
