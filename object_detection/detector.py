import ssl
import asyncio
import io
import asyncio
import io
import os
import sys

from PIL import Image
import google.generativeai as genai

from frame_msg import FrameMsg
from frame_msg.tx_capture_settings import TxCaptureSettings
from frame_msg.rx_photo import RxPhoto
from frame_msg.rx_tap import RxTap

class ObjectDetector:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY environment variable not found.")
            print("Please run: export GEMINI_API_KEY='your_api_key_here'")
            sys.exit(1)
            
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel('models/gemini-1.5-flash')
        
        self.frame = FrameMsg()
        # Create handlers
        self.rx_photo = RxPhoto()
        self.rx_tap = RxTap()
        
    def _print_handler(self, text):
        print(f"[Frame Lua]: {text}")

    async def update_display(self, text):
        try:
            await self.frame.print_short_text(text[:24]) # limited width available on screen
        except Exception as e:
            print(f"Failed to update display: {e}")

    async def run(self):
        print("Connecting to Frame...")
        try:
            await self.frame.connect()
            print("Connected! Initializing object detection app...")
            
            print("Uploading standard libraries...")
            await self.frame.upload_stdlua_libs(lib_names=['data', 'camera', 'tap'])

            print("Uploading camera app...")
            await self.frame.upload_frame_app(local_filename="lua/camera_frame_app.lua")

            self.frame.attach_print_response_handler(self._print_handler)
            
            image_queue = await self.rx_photo.attach(self.frame)
            
            tap_queue = await self.rx_tap.attach(self.frame)

            print("Starting Lua app on Frame...")
            await self.frame.start_frame_app()
            
            await asyncio.sleep(2)

            print("\n=================================")
            print(" Google Gemini Detection Ready!")
            print(" TAP the right side of your glasses to take a picture!")
            print(" Press Ctrl+C to exit")
            print("=================================\n")

            while True:
                #Waits for the user to tap the glasses
                tap_count = await tap_queue.get()
                print(f"Tap gesture detected! ({tap_count} taps)")
                
                await self.update_display("Taking photo...")
                
                #res for fast BT speeds (256x256), MEDIUM quality
                settings = TxCaptureSettings(resolution=256, quality_index=2, pan=0, raw=False)
                print("Triggering photo capture...")
                
                # 0x0d is the standard message code for TxCaptureSettings
                await self.frame.send_message(0x0d, settings.pack())
                
                try:
                    print("Waiting for image to download via Bluetooth...")
                    # We timeout after 10s if the transfer gets stuck
                    jpeg_bytes = await asyncio.wait_for(image_queue.get(), timeout=10.0)
                    
                    print("Photo received! Running Gemini AI...")
                    await self.update_display("Analyzing...")
                    
                    img = Image.open(io.BytesIO(jpeg_bytes))
                    
                    # Runs Gemini vision detection
                    prompt = (
                        "The camera is unfortunately VERY low quality but the user is counting on you to interpret the "
                        "blurry, pixelated images. NEVER comment on image quality. Do your best with images. "
                        "Identify the primary object or scene in ONE short phrase (max 3 words). "
                        "Example outputs: 'A coffee mug', 'A keyboard', 'Looking outside', 'A person'"
                    )
                    
                    # Call Gemini API
                    response = self.model.generate_content([prompt, img])
                    best_label = response.text.strip().replace('"', '').replace('.', '')
                    
                    print(f"✅ AI Found: {best_label}")

                    display_text = best_label if best_label else "Nothing found"
                    await self.update_display(display_text)
                    
                except asyncio.TimeoutError:
                    print("Photo download timed out. Retrying...")
                    await self.update_display("Retrying...")
                except Exception as e:
                    print(f"Gemini API Error: {e}")
                    await self.update_display("AI Error")
                
                await asyncio.sleep(2.0)

        except KeyboardInterrupt:
            print("\nShutting down cleanly...")
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Stopping app and disconnecting...")
            try:
                await self.frame.stop_frame_app()
                self.frame.detach_print_response_handler()
                self.rx_photo.detach(self.frame)
                self.rx_tap.detach(self.frame)
            except Exception:
                pass
            
            await self.frame.disconnect()
            print("Disconnected.")

if __name__ == "__main__":
    detector = ObjectDetector()
    try:
        asyncio.run(detector.run())
    except KeyboardInterrupt:
        pass
