@echo off
echo Installing dependencies and building for backend...
cd SDN-sim-mod-backend
start /wait cmd /c "npm install && npm run build"
echo Dependencies for backend installed and backend built successfully.

echo Installing dependencies and building for frontend...
cd ../SDN-sim-mod-frontend
start /wait cmd /c "npm install && npm run build"
echo Dependencies for frontend installed and frontend built successfully.

echo All packages installed and built. Closing all console windows...
taskkill /F /IM cmd.exe

echo All console windows closed.
