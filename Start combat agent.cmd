@echo off
cd /d "%~dp0"
py -3 agent.py play
if errorlevel 1 pause
