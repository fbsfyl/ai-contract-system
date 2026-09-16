﻿﻿﻿# 一键启动 AI 合同系统全部服务（FastAPI + Odoo 16）
# 用法：在项目根目录执行  .\start_all.ps1
# 说明：Odoo 与 FastAPI 均后台运行，关窗口不影响；日志见 logs\ 目录
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 确保日志目录存在
New-Item -ItemType Directory -Force -Path logs | Out-Null

Write-Host "==> 1/5 检查环境变量 .env" -ForegroundColor Cyan
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "    已生成 .env，请先填入 DEEPSEEK_API_KEY 后重新运行。" -ForegroundColor Yellow
    exit 1
}

Write-Host "==> 2/5 准备虚拟环境" -ForegroundColor Cyan
if (-not (Test-Path .venv)) {
    python -m venv .venv
}
$py = ".venv\Scripts\python.exe"

& $py -c "import fastapi, chromadb, sentence_transformers" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "    安装依赖..." -ForegroundColor Yellow
    & $py -m pip install -r requirements.txt
}

Write-Host "==> 3/5 写入向量库范例（RAG 检索库）" -ForegroundColor Cyan
& $py -B scripts\seed_examples.py

Write-Host "==> 4/5 启动 Odoo（docker compose，后台）" -ForegroundColor Cyan
docker compose -f docker-compose.odoo.yml up -d

Write-Host "==> 5/5 后台启动 FastAPI（0.0.0.0:8000）" -ForegroundColor Cyan
$existing = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "    8000 端口已占用，跳过启动 FastAPI" -ForegroundColor Yellow
} else {
    Start-Process -FilePath $py -ArgumentList @("-m","uvicorn","app.main:app","--host","0.0.0.0","--port","8000") -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput "logs\fastapi_console.log" -RedirectStandardError "logs\fastapi_error.log"
    Write-Host "    FastAPI 已在后台启动" -ForegroundColor Green
}

Write-Host ""
Write-Host "全部服务启动完成：" -ForegroundColor Green
Write-Host "  FastAPI : http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "  Odoo    : http://localhost:8069  （admin / admin）" -ForegroundColor Green
Write-Host "  容器内 Odoo 访问 AI 服务地址：http://host.docker.internal:8000" -ForegroundColor Yellow
Write-Host ""
Write-Host "  查看实时日志（另开窗口）：" -ForegroundColor Cyan
Write-Host "    Get-Content logs\app.log -Wait          # 业务 + HTTP 请求日志（主日志，UTF-8 带 BOM）" -ForegroundColor Cyan
Write-Host "    Get-Content logs\fastapi_error.log -Wait  # uvicorn 进程 stderr（启动报错/崩溃时看）" -ForegroundColor Cyan
Write-Host "  一键关闭全部服务：.\stop_all.ps1" -ForegroundColor Cyan
