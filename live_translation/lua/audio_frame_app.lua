-- audio_frame_app.lua
-- Uses the frame_msg audio library (audio.min) which wraps the correct
-- Lua microphone API: frame.microphone.start/read/stop, frame.bluetooth.send.
-- RxAudio(streaming=True) on the Python side expects 0x05/0x06 prefixed chunks.

local audio = require("audio.min")

-- Ping: Python calls this to verify the app is alive and start_record exists
function _G.ping()
    local tf = tostring(_G.start_record ~= nil)
    print("PING_RESP:ALIVE:TOGGLE_FUNC=" .. tf)
end

-- start_record(seconds): called by Python via send_lua()
-- Streams audio chunks to Python for the requested duration, then sends 0x06.
function _G.start_record(seconds)
    seconds = seconds or 5
    print("RECORD:STARTING")

    -- Start FPGA microphone: 8kHz, 8-bit matches RxAudio.to_wav_bytes() defaults
    audio.start({sample_rate=8000, bit_depth=8})

    -- Stream audio for the requested duration.
    -- frame.sleep() is the most reliable timing primitive available.
    -- We read in bursts to keep up with the mic; sleep(0) yields to BT stack.
    local stop_time = frame.time.utc() + seconds
    while frame.time.utc() < stop_time do
        audio.read_and_send_audio()
        frame.sleep(0)  -- yield to Bluetooth stack between reads
    end

    -- Stop the mic; next read() returns nil which sends the 0x06 FINAL sentinel
    audio.stop()
    frame.sleep(0.05)
    audio.read_and_send_audio()  -- flushes nil → sends FINAL (0x06) to Python

    print("RECORD:DONE")
end

-- Init: show ready state and return immediately so the interpreter
-- stays free to handle send_lua() calls (ping, start_record, etc.)
frame.display.text("Translator Ready", 50, 100)
frame.display.show()
print("APP:READY")
