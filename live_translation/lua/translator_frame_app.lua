local data = require('data.min')

USER_TEXT = 0x30  -- Message code for text messages

-- Function to scroll text horizontally
local function scroll_text(msg)
    local width = frame.display.width
    local text_len = string.len(msg)

    -- Scroll text one character at a time
    for i = 1, text_len do
        frame.display.text(string.sub(msg, i), 1, 1)
        frame.display.show()
        frame.sleep(0.2)
    end
end

-- Register parser
data.parsers[USER_TEXT] = function(msg)
    if string.len(msg) <= 16 then  -- Adjust 16 for your screen width
        frame.display.text(msg, 1, 1)
        frame.display.show()
    else
        scroll_text(msg)
    end
end

-- Main loop
function app_loop()
    print('Translator app running')
    while true do
        data.process_raw_items()
        frame.sleep(0.001)
    end
end

app_loop()
