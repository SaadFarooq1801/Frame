import asyncio
import io
from frame_sdk import Frame
from openai import AsyncOpenAI

client = AsyncOpenAI()

# Configuration
TARGET_LANGUAGE = "English"  # Change this to your preferred target language
AUTO_DETECT_SOURCE = True    # Auto-detect source language
SOURCE_LANGUAGE = "Spanish"  # Only used if AUTO_DETECT_SOURCE is False

async def display_text_scroll(frame, text, scroll_delay=1.5):
    """
    Displays long text by scrolling with word wrapping.
    """
    max_width = 120
    lines = []
    
    # Word wrap
    words = text.split()
    current_line = ""
    
    for word in words:
        if len(current_line) + len(word) + 1 <= max_width:
            current_line += word + " "
        else:
            if current_line:
                lines.append(current_line.strip())
            current_line = word + " "
    
    if current_line:
        lines.append(current_line.strip())
    
    # Display each line
    for i, line in enumerate(lines):
        await frame.display.set_text(line)
        await frame.display.show()
        
        # Shorter delay for last line
        if i < len(lines) - 1:
            await asyncio.sleep(scroll_delay)
        else:
            await asyncio.sleep(scroll_delay * 2)

async def transcribe_audio(audio_data):
    """
    Transcribe audio using OpenAI Whisper API with proper WAV formatting.
    """
    try:
        wav_bytes = audio_data.astype("int16").tobytes()
        audio_file = io.BytesIO()
        
        # WAV header configuration
        sample_rate = 16000
        num_channels = 1
        bits_per_sample = 16
        
        # Write WAV header
        audio_file.write(b'RIFF')
        audio_file.write((36 + len(wav_bytes)).to_bytes(4, 'little'))
        audio_file.write(b'WAVE')
        audio_file.write(b'fmt ')
        audio_file.write((16).to_bytes(4, 'little'))
        audio_file.write((1).to_bytes(2, 'little'))
        audio_file.write(num_channels.to_bytes(2, 'little'))
        audio_file.write(sample_rate.to_bytes(4, 'little'))
        audio_file.write((sample_rate * num_channels * bits_per_sample // 8).to_bytes(4, 'little'))
        audio_file.write((num_channels * bits_per_sample // 8).to_bytes(2, 'little'))
        audio_file.write(bits_per_sample.to_bytes(2, 'little'))
        audio_file.write(b'data')
        audio_file.write(len(wav_bytes).to_bytes(4, 'little'))
        audio_file.write(wav_bytes)
        
        audio_file.seek(0)
        
        # Transcribe with language detection
        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=("speech.wav", audio_file.read(), "audio/wav"),
            response_format="verbose_json"
        )
        
        detected_language = response.language if hasattr(response, 'language') else "unknown"
        text = response.text.strip()
        
        return text, detected_language
    
    except Exception as e:
        print(f"Transcription error: {e}")
        return "", "unknown"

async def translate_text(text, source_lang, target_lang):
    """
    Translate text using OpenAI's chat API with language awareness.
    """
    try:
        if AUTO_DETECT_SOURCE:
            prompt = f"Translate the following text to {target_lang}. Keep it concise and natural."
        else:
            prompt = f"Translate the following {source_lang} text to {target_lang}. Keep it concise and natural."
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator. {prompt} Provide ONLY the translation with no explanations or additional text."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Translation error: {e}")
        return None

async def main():
    async with Frame() as f:
        print("=" * 50)
        print("FRAME LIVE TRANSLATOR")
        print("=" * 50)
        print(f"Target Language: {TARGET_LANGUAGE}")
        print(f"Source Language: {'Auto-detect' if AUTO_DETECT_SOURCE else SOURCE_LANGUAGE}")
        print("Press Ctrl+C to stop")
        print("=" * 50)
        
        await f.display.set_text("Translator Ready")
        await f.display.show()
        
        try:
            while True:
                print("\n🎤 Listening...")
                
                # Record audio from Frame
                audio = await f.microphone.record_audio(
                    silence_cutoff_length_in_seconds=1.5,
                    max_length_in_seconds=10
                )
                
                if len(audio) == 0:
                    continue
                
                print(f"📊 Audio captured ({len(audio)} samples)")
                
                # Show processing status
                await f.display.set_text("Processing...")
                await f.display.show()
                
                # Transcribe
                text, detected_lang = await transcribe_audio(audio)
                
                if text == "":
                    await f.display.set_text("No speech")
                    await f.display.show()
                    await asyncio.sleep(1)
                    continue
                
                print(f"🗣️  Heard ({detected_lang}): {text}")
                
                # Skip translation if already in target language
                if detected_lang.lower() == TARGET_LANGUAGE.lower() or \
                   detected_lang.lower() == TARGET_LANGUAGE[:2].lower():
                    print(f"✓ Already in {TARGET_LANGUAGE}")
                    await display_text_scroll(f, text)
                    continue
                
                # Translate
                await f.display.set_text("Translating...")
                await f.display.show()
                
                translated = await translate_text(
                    text, 
                    detected_lang if AUTO_DETECT_SOURCE else SOURCE_LANGUAGE,
                    TARGET_LANGUAGE
                )
                
                if translated is None:
                    await f.display.set_text("Translation failed")
                    await f.display.show()
                    await asyncio.sleep(1)
                    continue
                
                print(f"🌍 Translated: {translated}")
                
                # Display on Frame
                await display_text_scroll(f, translated)
                
                # Pause before next recording
                await asyncio.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\n\n👋 Stopping translator...")
            await f.display.set_text("Goodbye!")
            await f.display.show()
            await asyncio.sleep(1)
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            await f.display.set_text("Error occurred")
            await f.display.show()
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
