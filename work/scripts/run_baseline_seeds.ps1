# Seeds 1-2 of the fixed baselines (seed 0 exists), identical protocol to seed 0; each followed by the
# full TEST evaluation (baselines are fixed, so no design decision uses these test numbers).
# Workers 5 + 2 for every new run (worker count changes speed only; data order is seeded per item).
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..
$logs = "E:\echo_project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
function Step($name, [scriptblock]$cmd) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $name" | Tee-Object -Append "$logs\run_baseline_seeds.log"
    & $cmd 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 "$logs\$name.log"
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] END $name exit=$LASTEXITCODE" | Tee-Object -Append "$logs\run_baseline_seeds.log"
}
$common = @('--T', '32', '--stride', '2', '--pad', 'loop', '--pool-valid', '--augment', '--batch', '16',
            '--epochs', '25', '--workers', '5', '--val-workers', '2')
foreach ($seed in 1, 2) {
    Step "train_bigru_s$seed" { .\py.cmd scripts\train.py --run "bigru_T32_s$seed" --temporal bigru --n-layers 3 @common --seed $seed }
    Step "eval_bigru_s$seed" { .\py.cmd scripts\evaluate.py --run "bigru_T32_s$seed" --workers 4 }
    Step "train_mamba_s$seed" { .\py.cmd scripts\train.py --run "mamba_T32_s$seed" --temporal mamba --n-layers 2 @common --seed $seed --temporal-kw 'expand=1' }
    Step "eval_mamba_s$seed" { .\py.cmd scripts\evaluate.py --run "mamba_T32_s$seed" --workers 4 }
    Step "train_uvt_s$seed" { .\py.cmd scripts\train.py --run "uvt_T32_s$seed" --temporal uvt --n-layers 2 @common --seed $seed }
    Step "eval_uvt_s$seed" { .\py.cmd scripts\evaluate.py --run "uvt_T32_s$seed" --workers 4 }
}
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ALL DONE" | Tee-Object -Append "$logs\run_baseline_seeds.log"
