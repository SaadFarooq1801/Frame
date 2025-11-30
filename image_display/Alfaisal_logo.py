import asyncio
from pathlib import Path
from frame_msg import FrameMsg, TxSprite 

async def main():
    frame = FrameMsg()

    try:
        await frame.connect()
        await frame.print_short_text("Loading Alfaisal Logo...")

        await frame.upload_stdlua_libs(lib_names=["data", "sprite"])
        await frame.upload_frame_app("lua/sprite_frame_app.lua")

        frame.attach_print_response_handler()
        await frame.start_frame_app()

        sprite = TxSprite.from_image_bytes(Path("images/alfaisal.png").read_bytes())
        await frame.send_message(0x20, sprite.pack())

        await asyncio.sleep(5)
        await frame.stop_frame_app()

    except Exception as e:
        print("Error:", e)

    finally:
        await frame.disconnect()

asyncio.run(main())
