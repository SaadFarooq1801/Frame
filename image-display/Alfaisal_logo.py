import asyncio
from pathlib import Path
from frame_msg import FrameMsg, TxSprite

async def main():
    frame = FrameMsg()
    try:
        await frame.connect()

        # Optional: check battery/memory
        batt_mem = await frame.send_lua('print(frame.battery_level() .. " / " .. collectgarbage("count"))', await_print=True)
        print(f"Battery Level/Memory used: {batt_mem}")

        # Show loading text
        await frame.print_short_text('Loading...')

        # Upload Lua libraries and main app
        await frame.upload_stdlua_libs(lib_names=['data', 'sprite'])
        await frame.upload_frame_app(local_filename="lua/sprite_frame_app.lua")
        frame.attach_print_response_handler()
        await frame.start_frame_app()

        # Load your logo image, quantize, and send
        image_path = Path("images/alfaisal_logo.jpg")  # <-- your logo here
        sprite = TxSprite.from_image_bytes(image_path.read_bytes())
        await frame.send_message(0x20, sprite.pack())

        # Display the logo for 5 seconds
        await asyncio.sleep(10.0)

        # Cleanup
        frame.detach_print_response_handler()
        await frame.stop_frame_app()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await frame.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
