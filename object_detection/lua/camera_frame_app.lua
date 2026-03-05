-- camera_frame_app.lua
-- Frameside app for object detection.
-- Relies on the standard `camera.min`, `data.min` and `tap.min` libraries

local data = require('data.min')
local camera = require('camera.min')
local tap = require('tap.min')

-- Message code for triggering capture from Python (must match tx_capture_settings.py msgCode 0x0d)
local CAPTURE_MSG = 0x0d

-- Override the capture settings parser so it executes immediately when the Python host sends TxCaptureSettings
data.parsers[CAPTURE_MSG] = function(msg)
    local settings = camera.parse_capture_settings(msg)
    -- Fire and forget the capture
    camera.capture_and_send(settings)
    -- Clear out the app data so it doesn't get processed again
    return nil
end

function app_loop()
    frame.display.text('Camera Ready', 1, 1)
    frame.display.show()

    -- Let the python script know we're ready
    print('Camera app is running')

    while true do
        rc, err = pcall(function()
            -- Wait for host commands
            local items_ready = data.process_raw_items()

            -- Run the auto exposure loop continuously so the camera is always ready
            if camera.is_auto_exp then
                camera.run_auto_exposure()
            end

            frame.sleep(0.1)
        end)

        if rc == false then
            print(err)
            break
        end
    end
end

app_loop()
