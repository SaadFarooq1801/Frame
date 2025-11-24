from PIL import Image

FRAME_WIDTH = 256
FRAME_HEIGHT = 128

def load_and_resize(path):
    img = Image.open(path).convert("1")  # 1-bit mode for Frame
    img = img.resize((FRAME_WIDTH, FRAME_HEIGHT))
    return img

def image_to_bytes(image):
    return image.tobytes()
