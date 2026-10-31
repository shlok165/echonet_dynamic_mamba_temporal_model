# Frozen PanEcho features (cache_features.py): pick ONE shared learning rate on VAL for all temporal
# blocks (seed 0, 40 epochs; BiGRU and the sync block at lr 1e-4, 3e-4, 1e-3). No test evaluation.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..
$logs = "E:\echo_project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
function Step($name, [scriptblock]$cmd) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $name" | Tee-Object -Append "$logs\run_feat_lr.log"
    & $cmd 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 "$logs\$name.log"
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] END $name exit=$LASTEXITCODE" | Tee-Object -Append "$logs\run_feat_lr.log"
}
$common = @('--features', 'panecho', '--T', '32', '--stride', '2', '--pad', 'loop', '--pool-valid', '--batch', '16',
            '--epochs', '40', '--seed', '0', '--workers', '2', '--val-workers', '1')
foreach ($lr in '1e-4', '3e-4', '1e-3') {
    Step "pe_bigru_lr$lr" { .\py.cmd scripts\train.py --run "pe_bigru_lr${lr}_s0" --temporal bigru --n-layers 3 @common --lr $lr }
    Step "pe_sync_lr$lr" { .\py.cmd scripts\train.py --run "pe_sync_lr${lr}_s0" --temporal sync --n-layers 2 @common --lr $lr --temporal-kw 'clock=hr,vol_readout=True,aux_weight=1.0,conv_pad=replicate,vol_trim=2' }
}
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ALL DONE" | Tee-Object -Append "$logs\run_feat_lr.log"
