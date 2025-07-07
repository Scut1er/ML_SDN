@echo off
echo Starting ML Balancer Service...
cd ml-balancer-service

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    start "" cmd /c "venv\Scripts\activate && python run.py run"
) else (
    echo Using global Python...
    start "" cmd /c "python run.py run"
)
echo ML Balancer Service started on port 5001.
echo Waiting 3 seconds for ML service to initialize...
timeout /t 3 /nobreak >nul

echo Building and starting the backend...
cd ../SDN-sim-mod-backend
start "" cmd /c "npm run build && npm run start"
echo Backend started.
echo Waiting 5 seconds for backend to build and start...
timeout /t 5 /nobreak >nul

echo Starting the frontend...
cd ../SDN-sim-mod-frontend
start "" cmd /c "npm run preview""
echo Frontend started.

cd ..
echo All services started:
echo - ML Balancer Service: http://localhost:5001
echo - Backend: http://localhost:3000
echo - Frontend: http://localhost:4173
echo.
echo Press any key to stop all services...
pause >nul

echo Stopping all services...
taskkill /F /IM python.exe
taskkill /F /IM node.exe

echo All services stopped. 