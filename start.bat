@echo off
chcp 65001 >nul
echo ============================================================
echo   信贷材料预审 Agent · 一键启动
echo ============================================================
echo.

cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

echo [1/3] 检查依赖...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo   正在安装依赖...
    pip install -r requirements.txt
)

echo [2/3] 检查配置...
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo   已创建 .env，请填入 API Key 后继续（无Key也可启动，仅本地规则可用）
        echo.
        timeout /t 2 >nul
    )
)

echo [3/3] 启动服务...
echo.
echo ============================================================
echo   上传门户: http://127.0.0.1:8765/portal
echo ============================================================
echo.
python run.py web

pause
