import asyncio
from pathlib import Path
from frame_msg import FrameMsg, TxSprite

async def main():
    """
    Display an image directly on the Frame display using TxSprite for 16-color quantization.
    """

    frame = FrameMsg()
    try:
        await frame.connect()

        # Optional: check battery and memory
        batt_mem = await frame.send_lua('print(frame.battery_level() .. " / " .. collectgarbage("count"))', await_print=True)
        print(f"Battery Level/Memory used: {batt_mem}")

        # Show loading text
        await frame.print_short_text('Loading...')

        # Upload necessary Lua libraries for sprite handling
        await frame.upload_stdlua_libs(lib_names=['data', 'sprite'])

        # Upload the main Lua app that handles sprites
        await frame.upload_frame_app(local_filename="lua/sprite_frame_app.lua")

        # Attach handler for Lua print statements
        frame.attach_print_response_handler()

        # Start the Frameside app
        await frame.start_frame_app()

        # Load image bytes, quantize to 16 colors, and send to Frame
        image_path = Path("images/koala.jpg")  # <-- replace with your image
        sprite = TxSprite.from_image_bytes(image_path.read_bytes())
        await frame.send_message(0x20, sprite.pack())

        # Display image for 5 seconds
        await asyncio.sleep(5.0)

        # Cleanup
        frame.detach_print_response_handler()
        await frame.stop_frame_app()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await frame.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
