# 一键启动 AI 合同系统（Windows PowerShell）
# 用法：在项目根目录执行  .\start.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==> 1/4 检查环境变量 .env" -ForegroundColor Cyan
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "    已生成 .env，请先填入 DEEPSEEK_API_KEY 后重新运行。" -ForegroundColor Yellow
    exit 1
}

Write-Host "==> 2/4 准备虚拟环境" -ForegroundColor Cyan
if (-not (Test-Path .venv)) {
    python -m venv .venv
}

$py = ".venv\Scripts\python.exe"

# 依赖未安装则安装（首次较慢，会拉取 torch / chromadb / opencv 等）
& $py -c "import fastapi, chromadb, sentence_transformers" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "    安装依赖..." -ForegroundColor Yellow
    & $py -m pip install -r requirements.txt
}

Write-Host "==> 3/4 写入向量库范例（RAG 检索库）" -ForegroundColor Cyan
& $py -B scripts\seed_examples.py

Write-Host "==> 4/4 启动服务 http://127.0.0.1:8000" -ForegroundColor Green
& $py -m uvicorn app.main:app --host 127.0.0.1 --port 8000
