-- frame_app.lua
-- Minimal Frame Lua app for displaying text

-- Message code for text from Python
USER_TEXT = 0x20

-- Require built-in Frame libraries
local data = require("data.min")   -- for parsing messages
local sprite = require("sprite.min")

-- Register parser for our message code
data.parsers[USER_TEXT] = sprite.parse_sprite

-- Function to display text on Frame
function display_text(text)
    frame.display.text("                ", 1, 1)  -- clear line
    frame.display.text(text, 1, 1)
    frame.display.show()
end

-- Main loop
function app_loop()
    print("Frame app running")  -- signals Python that app is ready

    while true do
        local rc, err = pcall(function()
            local items_ready = data.process_raw_items()
            
            if items_ready > 0 and data.app_data[USER_TEXT] then
                local msg = data.app_data[USER_TEXT]
                local text = msg.pixel_data or ""  -- payload from Python

                -- Display text
                display_text(text)

                -- Clear for next message
                data.app_data[USER_TEXT] = nil
                collectgarbage()
            end
            frame.sleep(0.001)
        end)

        if rc == false then
            print("Error in loop:", err)
        end
    end
end

-- Start main loop
app_loop()
