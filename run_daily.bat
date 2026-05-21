@echo off
chcp 65001 >nul
cd /d "D:\All code\quant_system"
echo ==========================================
echo [1/3] Update daily bars
echo ==========================================
python run_daily_update_v2.py
if errorlevel 1 goto :error

echo.
echo ==========================================
echo [2/3] Scan market signals
echo ==========================================
python run_signal_scan.py
if errorlevel 1 goto :error

echo.
echo ==========================================
echo [3/3] Save stats to xlsx
echo ==========================================
python run_save_stats.py
if errorlevel 1 goto :error

echo.
echo ==========================================
echo All done.
echo ==========================================
pause
exit /b 0

:error
echo.
echo ==========================================
echo FAILED. Check logs/ for details.
echo ==========================================
pause
exit /b 1
