import asyncio
from pathlib import Path
from frame_sdk import Frame
from frame_sdk.txsprite import TxSprite

async def main():
    async with Frame() as f:
        # Load image
        image_bytes = Path("images/alfaisal_logo.jpg").read_bytes()
        sprite = TxSprite.from_image_bytes(image_bytes)

        # Send to Frame
        await f.send_message(0x20, sprite.pack())

asyncio.run(main())
