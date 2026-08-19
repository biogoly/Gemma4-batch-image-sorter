@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" sort_images.py ^
    --input-dir input ^
    --output-dir output ^
    --feature-file prompts\feature.txt ^
    --base-url http://127.0.0.1:8080/v1 ^
    --model local-model ^
    --mode copy ^
    --recursive ^
    --run-label gemma-test

echo.
pause
