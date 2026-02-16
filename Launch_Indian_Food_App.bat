@echo off
echo ============================================
echo    Food Recipe Generator
echo    Indian + Western Cuisine
echo ============================================
echo.
echo Starting the web app...
echo.

cd /d "%~dp0"
cd inversecooking\src

"..\..\.venv\Scripts\python.exe" web_app_combined.py

pause
