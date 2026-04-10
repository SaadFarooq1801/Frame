# Frame
Programs for Frame by Brillant Labs

In this repo you will find 4 different programs:
- Image display
- Msg display
- Live translation
- Object detection

live translation has only been tested on mac, testing on window devices still in progress. (It may or may not work)

lua is the main programming language frame, it allows direct communication with the frame’s hardware peripherals.
Make sure lua files/folder are present when running "image display", "live_translation", and "object_detection".

--------------------------------------------------------------------------
## Live Translation
Live translation uses **OpenAI Whisper** for on-device speech recognition (no API key needed).

### First-Run Model Download
On the first run, Whisper will automatically download the `small` model (~460MB):
The model is cached at `~/.cache/whisper/small.pt` and **never downloaded again** on the same machine.

**Note for corporate/university networks:** If you get an SSL certificate error on first run, this is caused by a network proxy. The code already handles this automatically.

### Whisper Models
You can change the model in `translator_mac.py` to trade off speed vs accuracy:

| Model | Size | Best for |
|-------|------|----------|
| `tiny` | 75MB | Testing only |
| `base` | 150MB | Fast, low accuracy |
| `small` | 460MB | ✅ Recommended |
| `medium` | 1.5GB | High accuracy, slower |
| `large` | 3GB | Best accuracy |

--------------------------------------------------------------------------
## Object Detection
The object detection module uses the Frame camera to stream JPEG photos over Bluetooth to your Mac/PC, where it runs **YOLOv8** (nano) to detect objects in real-time. The top detected label is then sent back to the Frame display.

### Setup
1. `cd object_detection/`
2. `pip install -r requirements.txt` (inside your venv)
3. Connect your Frame via Bluetooth.
4. Run `python detector.py`

*Note: The YOLOv8 nano model (`yolov8n.pt`) is downloaded automatically on the first run (~6 MB).*

----------------------------------------------------------

Link for Frame Libraries:
https://docs.brilliant.xyz/frame/frame-sdk-lua/



------------------------------------------------------------------------------------------------------
## Setup
clone this repo onto your device and run in a virtual enviroment.

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

deactivate (Once your done)
