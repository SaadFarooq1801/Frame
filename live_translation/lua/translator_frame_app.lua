local data = require('data.min')
local audio = require('audio.min')
local code  = require('code.min')

-- Message codes
AUDIO_CTRL = 0x30      -- start/stop microphone
TEXT_OUT   = 0x20      -- text from Python to display

-- Register parser so incoming text messages are handled
data.parsers[TEXT_OUT] = function(msg)
    local txt = msg.raw or ""
    frame.display.clear()
    frame.display.text(txt, 1, 1)
    frame.display.show()
end

function app_loop()
    frame.display.clear()
    frame.display.text("Translator Ready", 1, 1)
    frame.display.show()

    print("Frame Lua app running")

    while true do
        local ok, err = pcall(function()

            -- Process incoming data (text from Python)
            local items_ready = data.process_raw_items()
            if items_ready > 0 then
                -- handled automatically by parser
            end

            -- If microphone is enabled, audio.min will automatically
            -- produce audio samples that go to the host via data channel
            frame.sleep(0.001)

        end)

        if not ok then
            print(err)
            break
        end
    end
end

app_loop()
