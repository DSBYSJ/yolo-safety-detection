@echo off
REM 一键启动安全帽/口罩佩戴检测系统（Windows）
REM 用法：双击本文件，或在项目根目录执行  scripts\run_web.bat

setlocal
cd /d "%~dp0.."

REM 使用已装好 ultralytics + torch(CUDA) 的 Python 环境
set PY=C:\Users\20261\.workbuddy\binaries\python\envs\default\Scripts\python.exe

REM 摄像头设备索引。本机实测：索引 0 打不开，真实摄像头在索引 1。
REM 换机器或插拔设备后索引可能变，先跑 scripts\list_cameras.py 确认。
REM 若改用网络流（IP Webcam 等），设 CAMERA_SOURCE 并注释掉本行即可。
set CAMERA_INDEX=1

if not exist "%PY%" (
    echo [ERROR] 未找到 Python 环境: %PY%
    echo         请修改本文件中的 PY 变量，指向已安装依赖的解释器。
    pause
    exit /b 1
)

echo ============================================================
echo   安全帽/口罩佩戴检测系统  ^|  YOLOv8 + Flask
echo   启动地址: http://127.0.0.1:5000
echo   停止服务: 在本窗口按 Ctrl+C
echo ============================================================
echo.

"%PY%" webapp\app.py

pause
