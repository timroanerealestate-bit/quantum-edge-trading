@echo off
title Training Bot Dashboard
color 0A
echo.
echo  ================================================
echo   Training Bot -- Institutional Signal Scanner
echo  ================================================
echo.
echo  Starting Streamlit dashboard...
echo  Your browser will open automatically.
echo  Press Ctrl+C in this window to stop the server.
echo.

cd /d "%~dp0"

where streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo  Streamlit not found on PATH -- trying Python module...
    python -m streamlit run dashboard.py --server.headless false --browser.gatherUsageStats false
) else (
    streamlit run dashboard.py --server.headless false --browser.gatherUsageStats false
)

echo.
echo  Server stopped.
pause
