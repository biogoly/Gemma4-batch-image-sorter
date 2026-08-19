@echo off
setlocal

rem EDIT THESE TWO PATHS. Quotation marks are required when paths contain spaces.
set "LLAMA_SERVER=C:\path\to\llama-server.exe"
set "MODEL=C:\path\to\gemma-model.gguf"
set "MMPROJ=C:\path\to\mmproj-model.gguf"

"%LLAMA_SERVER%" ^
    -m "%MODEL%" ^
    --mmproj "%MMPROJ%" ^
    --host 127.0.0.1 ^
    --port 8080 ^
    -ngl all ^
    -c 8192 ^
    -fa on ^
    -ts 1,1

pause
