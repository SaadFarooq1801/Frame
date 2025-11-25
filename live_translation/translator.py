import asyncio
import io
import wave
import numpy as np
from frame_sdk import Frame
from googletrans import Translator
import speech_recognition as sr

# Initialize Google Translator
translator = Translator()

# Configuration
TARGET_LANGUAGE = "en"  # Language codes: en, es, fr, de, ar, zh-cn, etc.

async def display_text_scroll(frame, text, scroll_delay=2.0):
   """
   Displays long text by scrolling with word wrapping.
   """
   max_width = 640
   lines = []
   
   # Word wrap
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
   
   # Display each line
   for i, line in enumerate(lines):
       await frame.display.show_text(line, 50, 100)
       
       if i < len(lines) - 1:
           await asyncio.sleep(scroll_delay)
       else:
           await asyncio.sleep(scroll_delay * 1.5)

def save_wav_file(audio_data, filename="temp_audio.wav", sample_rate=16000):
   """
   Save audio data as a WAV file for better compatibility.
   """
   try:
       # Ensure audio is in int16 format
       audio_int16 = audio_data.astype("int16")
       
       # Create WAV file
       with wave.open(filename, 'wb') as wav_file:
           wav_file.setnchannels(1)  # Mono
           wav_file.setsampwidth(2)  # 2 bytes per sample (int16)
           wav_file.setframerate(sample_rate)
           wav_file.writeframes(audio_int16.tobytes())
       
       return filename
   except Exception as e:
       print(f"Error saving WAV: {e}")
       return None

def normalize_audio(audio_data):
   """
   Normalize audio to prevent clipping and improve quality.
   """
   # Check if audio is clipping
   max_val = np.max(np.abs(audio_data))
   if max_val > 30000:  # If clipping detected
       print(f"⚠️  Audio clipping detected (max: {max_val}), normalizing...")
       # Reduce gain to prevent clipping
       reduction_factor = 0.7 * (30000 / max_val)  # Reduce to 70% of non-clipping level
       audio_data = audio_data * reduction_factor
   
   return audio_data

def transcribe_audio_local(audio_data, sample_rate=16000):
   """
   Transcribe audio using Google Speech Recognition.
   Tries multiple methods for better compatibility.
   """
   recognizer = sr.Recognizer()
   
   # More sensitive recognition settings for Frame microphone
   recognizer.energy_threshold = 100  # Lower threshold for quieter mics
   recognizer.dynamic_energy_threshold = True
   recognizer.pause_threshold = 1.0  # Slightly longer pause threshold
   
   # Normalize audio first to prevent clipping issues
   audio_data = normalize_audio(audio_data)
   
   try:
       # Method 1: Using AudioData directly
       print("Attempting transcription (method 1)...")
       audio_int16 = audio_data.astype("int16")
       audio_bytes = audio_int16.tobytes()
       audio = sr.AudioData(audio_bytes, sample_rate, 2)
       
       text = recognizer.recognize_google(audio)  # Auto-detect language
       return text.strip()
   
   except sr.UnknownValueError:
       print("Method 1 failed: Could not understand audio")
   except sr.RequestError as e:
       print(f"Method 1 failed: API error - {e}")
   except Exception as e:
       print(f"Method 1 failed: {e}")
   
   try:
       # Method 2: Save to WAV file first
       print("Attempting transcription (method 2 - via WAV file)...")
       wav_filename = save_wav_file(audio_data, sample_rate=sample_rate)
       
       if wav_filename:
           with sr.AudioFile(wav_filename) as source:
               audio = recognizer.record(source)
               text = recognizer.recognize_google(audio)
               return text.strip()
   
   except sr.UnknownValueError:
       print("Method 2 failed: Could not understand audio")
   except sr.RequestError as e:
       print(f"Method 2 failed: API error - {e}")
   except Exception as e:
       print(f"Method 2 failed: {e}")
   
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
   Detect the language of the text.
   """
   try:
       result = translator.detect(text)
       return result.lang
   except Exception as e:
       print(f"Language detection error: {e}")
       return "unknown"

def analyze_audio_quality(audio_data):
   """
   Analyze audio quality and provide feedback.
   """
   if len(audio_data) == 0:
       return "empty", 0
   
   rms = np.sqrt(np.mean(audio_data**2))
   max_val = np.max(np.abs(audio_data))
   
   print(f"🔊 Audio Analysis:")
   print(f"   - RMS (loudness): {rms:.2f}")
   print(f"   - Peak amplitude: {max_val:.2f}")
   print(f"   - Duration: {len(audio_data)/16000:.2f}s")
   
   # Adjusted thresholds for Frame microphone
   if max_val > 30000:
       return "clipping", rms
   elif max_val < 100:  # Increased threshold for quiet detection
       return "too_quiet", rms
   elif rms < 20:  # Use RMS for better quiet detection
       return "too_quiet", rms
   else:
       return "good", rms

async def main():
   async with Frame() as f:
       print("=" * 60)
       print("FRAME LIVE TRANSLATOR (Google Translate)")
       print("=" * 60)
       print(f"Target Language: {TARGET_LANGUAGE}")
       print("No API key required!")
       print("Press Ctrl+C to stop")
       print("=" * 60)
       
       await f.display.show_text("Translator Ready", 50, 100)
       
       try:
           recording_count = 0
           
           while True:
               recording_count += 1
               print(f"\n{'='*60}")
               print(f"🎤 Recording #{recording_count} - Speak now...")
               print(f"{'='*60}")
               
               # Show recording indicator
               await f.display.show_text("🎤 Recording...", 50, 100)
               
               # Record audio from Frame
               try:
                   audio = await f.microphone.record_audio(
                       silence_cutoff_length_in_seconds=1.5,
                       max_length_in_seconds=10
                   )
               except Exception as e:
                   print(f"❌ Recording error: {e}")
                   await f.display.show_text("Recording failed", 50, 100)
                   await asyncio.sleep(2)
                   continue
               
               if len(audio) == 0:
                   print("⚠️  No audio captured (empty buffer)")
                   await f.display.show_text("No audio detected", 50, 100)
                   await asyncio.sleep(1)
                   continue
               
               # Analyze audio quality
               quality, loudness = analyze_audio_quality(audio)
               
               if quality == "clipping":
                   print("⚠️  Audio clipping detected - speaking more softly or moving microphone further")
                   await f.display.show_text("Audio clipping - speak softer", 50, 100)
                   await asyncio.sleep(2)
                   # Continue anyway with normalized audio
                   print("🔄 Attempting transcription with normalized audio...")
               elif quality == "too_quiet":
                   print("⚠️  Audio too quiet - try speaking louder and closer to mic")
                   await f.display.show_text("Too quiet - speak louder", 50, 100)
                   await asyncio.sleep(2)
                   continue
               
               print(f"📊 Audio captured:")
               print(f"   - Samples: {len(audio)}")
               print(f"   - Duration: ~{len(audio) / 16000:.2f} seconds")
               print(f"   - Quality: {quality}")
               
               # Show processing status
               await f.display.show_text("Processing...", 50, 100)
               
               # Transcribe (audio normalization happens inside this function)
               text = transcribe_audio_local(audio)
               
               if text == "":
                   print("❌ No speech recognized")
                   await f.display.show_text("No speech detected", 50, 100)
                   await asyncio.sleep(2)
                   continue
               
               # Detect language
               detected_lang = detect_language(text)
               print(f"✅ Heard ({detected_lang}): '{text}'")
               
               # Skip translation if already in target language
               if detected_lang == TARGET_LANGUAGE:
                   print(f"✓ Already in target language")
                   await display_text_scroll(f, text)
                   continue
               
               # Translate
               await f.display.show_text("Translating...", 50, 100)
               
               translated = translate_text(text, TARGET_LANGUAGE)
               
               if translated is None or translated == "":
                   print("❌ Translation failed")
                   await f.display.show_text("Translation failed", 50, 100)
                   await asyncio.sleep(2)
                   continue
               
               print(f"🌍 Translated: '{translated}'")
               
               # Display on Frame
               await display_text_scroll(f, translated)
               
               # Pause before next recording
               print("\nReady for next recording...")
               await asyncio.sleep(1)
       
       except KeyboardInterrupt:
           print("\n\n👋 Stopping translator...")
           await f.display.show_text("Goodbye!", 50, 100)
           await asyncio.sleep(1)
       
       except Exception as e:
           print(f"\n❌ Error: {e}")
           import traceback
           traceback.print_exc()
           await f.display.show_text("Error occurred", 50, 100)
           await asyncio.sleep(2)

if __name__ == "__main__":
   asyncio.run(main())
