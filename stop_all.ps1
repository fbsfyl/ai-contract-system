# 一键关闭 AI 合同系统全部服务（FastAPI + Odoo 16）
# 用法：在项目根目录执行  .\stop_all.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==> 1/2 停止 FastAPI 后台进程（占用 8000 端口）" -ForegroundColor Cyan
$conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($conn) {
    $conn | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
        Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
        Write-Host "    已停止 FastAPI（PID $_）" -ForegroundColor Green
    }
} else {
    Write-Host "    FastAPI 未运行" -ForegroundColor Yellow
}

Write-Host "==> 2/2 停止 Odoo 容器" -ForegroundColor Cyan
docker compose -f docker-compose.odoo.yml down

Write-Host ""
Write-Host "全部服务已停止。" -ForegroundColor Green
