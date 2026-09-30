@echo off
REM ============================================================
REM  一键启动：安全帽/口罩佩戴检测系统（Windows）
REM
REM  用法：双击本文件，或在项目根目录执行  scripts\run_web.bat
REM  停止：在本窗口按 Ctrl+C，或直接关掉窗口
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0.."

REM ---------- 解释器 ----------
REM 必须是装好 ultralytics + torch(CUDA) 的那个环境。
REM 用系统 Python 会因缺 ultralytics 而启动失败。
REM
REM 自动探测顺序（找到就用，兜底才是本机写死路径）：
REM   1) 项目内虚拟环境  .venv\Scripts\python.exe
REM   2) 环境变量 PYTHON_BIN 指定的解释器
REM   3) PATH 里的 python
REM   4) 本机原来的固定路径（换机器后大概率不存在）
set PY=
if exist "%~dp0..\.venv\Scripts\python.exe" set PY=%~dp0..\.venv\Scripts\python.exe
if not defined PY if exist "%~dp0..\venv\Scripts\python.exe" set PY=%~dp0..\venv\Scripts\python.exe
if not defined PY if defined PYTHON_BIN if exist "%PYTHON_BIN%" set PY=%PYTHON_BIN%
if not defined PY for /f "delims=" %%p in ('where python 2^>nul') do if not defined PY set PY=%%p
if not defined PY set PY=C:\Users\20261\.workbuddy\binaries\python\envs\default\Scripts\python.exe

REM ---------- 取流来源 ----------
REM 本机实测：索引 0 打不开（MSMF: can't grab frame），真实摄像头在索引 1。
REM 换机器或插拔设备后索引会变，先跑 scripts\list_cameras.py 确认。
set CAMERA_INDEX=1

REM 若改用网络流（IP Webcam / RTSP），取消下面一行注释并改地址。
REM 设了它之后 CAMERA_INDEX 与分辨率设置都会被忽略。
REM set CAMERA_SOURCE=rtsp://admin:密码@192.168.1.100:554/Streaming/Channels/102

REM ---------- HTTPS ----------
REM 手机浏览器只有在 HTTPS 或 localhost 下才允许调用摄像头。
REM 只用电脑看的话可以注释掉这两行，改走 HTTP。
set FLASK_SSL_CERT=certs/cert.pem
set FLASK_SSL_KEY=certs/key.pem

REM ============================================================
REM  以下为启动逻辑，一般不需要改
REM ============================================================

if not exist "%PY%" (
    echo.
    echo [错误] 未找到可用的 Python 环境。
    echo.
    echo        已探测位置依次为：
    echo          1^) .venv\Scripts\python.exe
    echo          2^) 环境变量 PYTHON_BIN
    echo          3^) PATH 中的 python
    echo          4^) %PY%
    echo.
    echo        请在项目根目录创建虚拟环境并安装依赖：
    echo            python -m venv .venv
    echo            .venv\Scripts\pip install -r requirements.txt
    echo.
    echo        或设置环境变量 PYTHON_BIN 指向已装好依赖的解释器。
    echo.
    pause
    exit /b 1
)

REM ---------- 清理残留：端口被旧进程占着会导致新服务静默启动失败 ----------
set OLD_PID=
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING"') do set OLD_PID=%%p
if not "%OLD_PID%"=="" (
    echo [提示] 检测到 5000 端口被进程 %OLD_PID% 占用，正在结束它...
    taskkill /F /PID %OLD_PID% >nul 2>&1
    REM 等端口真正释放，否则新进程绑不上
    ping -n 3 127.0.0.1 >nul
    echo [提示] 旧进程已清理
)

REM ---------- 证书检查 ----------
set SCHEME=http
if exist "%FLASK_SSL_CERT%" if exist "%FLASK_SSL_KEY%" set SCHEME=https
if "%SCHEME%"=="https" goto show_banner
echo [警告] 未找到证书 %FLASK_SSL_CERT% / %FLASK_SSL_KEY%
echo        将以 HTTP 启动 —— 手机端将无法调用摄像头。
echo        生成证书的办法见 README「六之四」。
echo.

:show_banner
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr /c:"IPv4"') do (
    for /f "tokens=1" %%j in ("%%i") do if not defined LAN_IP set LAN_IP=%%j
)

echo ============================================================
echo   安全帽 / 口罩佩戴检测系统   YOLOv8 + Flask
echo ------------------------------------------------------------
echo   本机访问 : %SCHEME%://127.0.0.1:5000
echo   手机访问 : %SCHEME%://%LAN_IP%:5000
echo   停止服务 : 在本窗口按 Ctrl+C
echo ============================================================
echo.
if "%SCHEME%"=="https" (
    echo   提示：首次访问会提示证书不受信任，选「继续前往」即可。
    echo.
)

REM ---------- 自动打开浏览器（等 6 秒给服务留加载时间）----------
start "" cmd /c "ping -n 7 127.0.0.1 >nul & start %SCHEME%://127.0.0.1:5000"

REM ---------- 前台运行，这样关窗口就等于停服务 ----------
"%PY%" -u webapp\app.py

echo.
echo [服务已停止]
pause
