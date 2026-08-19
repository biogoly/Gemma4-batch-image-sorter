@echo off
setlocal
cd /d "%~dp0"

py -3.12 -m venv .venv
if errorlevel 1 goto :error

".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Setup complete.
pause
exit /b 0

:error
echo.
echo Setup failed. See the error above.
pause
exit /b 1
