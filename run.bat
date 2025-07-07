@echo off
echo Starting the backend...
cd SDN-sim-mod-backend
start "" cmd /c "npm run start"
echo Backend started.

echo Starting the frontend...
cd ../SDN-sim-mod-frontend
start "" cmd /c "npm run preview"
echo Frontend started.

cd ..
echo All processes started. Press any key to exit.
pause >nul

echo Stopping the backend...
taskkill /F /IM node.exe

echo Stopping the frontend...
taskkill /F /IM node.exe

echo All processes stopped.
