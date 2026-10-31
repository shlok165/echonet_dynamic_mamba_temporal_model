# Sequential UVT pipeline (one GPU job at a time). Logs go to E:\echo_project\logs\.
# 1) UVT-style temporal block in our protocol (EF-only, same settings as the BiGRU/Mamba runs) + full eval
# 2) original UVT recipe at reduced size (ED/ES + EF loss) + full-video eval
# 3) released paper-size UVT weights, full-video eval (if downloaded)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..
$logs = "E:\echo_project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
function Step($name, [scriptblock]$cmd) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $name" | Tee-Object -Append "$logs\run_uvt_all.log"
    & $cmd 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 "$logs\$name.log"
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] END $name exit=$LASTEXITCODE" | Tee-Object -Append "$logs\run_uvt_all.log"
}

# workers kept low: ~720 MB RAM per worker process and ~6 GB free; worker count affects speed only
Step "train_uvt_block" { .\py.cmd scripts\train.py --run uvt_T32_s0 --temporal uvt --n-layers 2 --T 32 --stride 2 --pad loop --pool-valid --augment --batch 16 --epochs 25 --seed 0 --workers 3 --val-workers 1 }
Step "eval_uvt_block" { .\py.cmd scripts\evaluate.py --run uvt_T32_s0 --workers 3 }
Step "train_uvt_orig_small" { .\py.cmd scripts\train_uvt.py --run uvt_orig_small_M_reg --preset small --mode repeat --sdmode reg --epochs 5 --seed 0 --workers 3 }
Step "eval_uvt_orig_small" { .\py.cmd scripts\eval_uvt.py --run uvt_orig_small_M_reg --workers 1 }
# downloads only after training (a ~4 GB stream next to training exhausted RAM once)
Step "download_M_REG" { .\py.cmd scripts\download_uvt_weights.py M REG }
Step "download_R_REG" { .\py.cmd scripts\download_uvt_weights.py R REG }
foreach ($k in "M_REG", "R_REG") {
    $w = "E:\echo_project\cache\uvt_weights\UVT_$k\best.pt"
    if (Test-Path $w) {
        Step "eval_uvt_released_$k" { .\py.cmd scripts\eval_uvt.py --released $w --sdmode reg --tag "released_$k" --workers 1 }
    } else {
        "released weights $k not found at $w; skipped" | Tee-Object -Append "$logs\run_uvt_all.log"
    }
}
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ALL DONE" | Tee-Object -Append "$logs\run_uvt_all.log"
