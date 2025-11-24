local data = require('data.min')

USER_TEXT = 0x30  -- Message code for text messages

-- Register parser
data.parsers[USER_TEXT] = function(msg)
    frame.display.text(msg, 1, 1)
    frame.display.show()
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
