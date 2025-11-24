from frame_sdk import Frame
import time
from image_loader import load_and_resize, image_to_bytes

frame = Frame()

def display_image(path):
    img = load_and_resize(path)
    data = image_to_bytes(img)

    frame.display.draw_bitmap(
        x=0,
        y=0,
        width=256,
        height=128,
        bitmap_bytes=data
    )

    frame.display.show()

if __name__ == "__main__":
    display_image("images/my_image.png")
    time.sleep(5)
