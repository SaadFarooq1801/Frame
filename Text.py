import asyncio
from frame_sdk import Frame

async def main():
    async with Frame() as f:
        await f.run_lua("frame.display.text('I am Saad', 50, 50, 100); frame.display.show()")

asyncio.run(main())
