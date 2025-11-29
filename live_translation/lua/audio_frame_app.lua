-- Audio Frame App for Live Translation
local data = require('data.min')
local audio = require('audio.min')
local code = require('code.min')

-- Message codes
local AUDIO_CTRL = 0x30

-- Global state
local streaming = false

-- Audio control handler
data.parsers[AUDIO_CTRL] = function(msg)
    local ctrl = code.parse_code(msg.data)
    print("Audio control received: " .. tostring(ctrl))
    
    if ctrl == 1 then
        -- Start audio streaming
        streaming = true
        audio.start_audio_stream()
        print("Audio stream STARTED")
    elseif ctrl == 0 then
        -- Stop audio streaming
        streaming = false
        audio.stop_audio_stream()
        print("Audio stream STOPPED")
    end
end

-- Main app loop
function app_loop()
    frame.display.clear()
    frame.display.text("Audio Ready", 50, 100)
    frame.display.show()
    
    -- Signal ready to Python
    print("ready")
    
    while true do
        local ok, err = pcall(function()
            -- Process incoming control messages
            local items = data.process_raw_items()
            
            -- If streaming, audio.min automatically sends samples
            -- via the data channel back to Python
            
            frame.sleep(0.01)
        end)
        
        if not ok then
            print("Error: " .. tostring(err))
            frame.sleep(1)
        end
    end
end

app_loop()
