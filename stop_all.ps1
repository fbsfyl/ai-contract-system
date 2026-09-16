﻿﻿﻿# 一键关闭 AI 合同系统全部服务（FastAPI + Odoo 16）
# 用法：在项目根目录执行  .\stop_all.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==> 1/2 停止 FastAPI 后台进程" -ForegroundColor Cyan
$killed = $false

# 杀掉所有命令行含 uvicorn 的 python 进程（含 venv 启动器 + 实际解释器这一对父子进程）
Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "uvicorn" } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "    已停止 uvicorn 进程（PID $($_.ProcessId)）" -ForegroundColor Green
        $killed = $true
    }

# 兜底：再按端口清一遍，防止命令行读取不到的漏网进程
$conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($conn) {
    $conn | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
        Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
        Write-Host "    已停止 8000 端口进程（PID $_）" -ForegroundColor Green
        $killed = $true
    }
}

if (-not $killed) {
    Write-Host "    FastAPI 未运行" -ForegroundColor Yellow
}

Write-Host "==> 2/2 停止 Odoo 容器" -ForegroundColor Cyan
docker compose -f docker-compose.odoo.yml down

Write-Host ""
Write-Host "全部服务已停止。" -ForegroundColor Green
