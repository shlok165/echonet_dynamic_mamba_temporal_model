# Pilot runs of the heart-rate-tuned SSM ("sync"), seed 0, judged on VAL only (no test evaluation).
# Same protocol as the baselines (T=32, stride 2, loop, pool-valid, augment, batch 16, 25 epochs).
#   A: clock + volumetric readout (+ EF-consistency loss)   = full proposal
#   C: no clock + volumetric readout                         -> effect of the clock
#   B: clock, no volumetric readout                          -> effect of the readout
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..
$logs = "E:\echo_project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
function Step($name, [scriptblock]$cmd) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $name" | Tee-Object -Append "$logs\run_sync_pilots.log"
    & $cmd 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 "$logs\$name.log"
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] END $name exit=$LASTEXITCODE" | Tee-Object -Append "$logs\run_sync_pilots.log"
}
$common = @('--temporal', 'sync', '--n-layers', '2', '--T', '32', '--stride', '2', '--pad', 'loop', '--pool-valid',
            '--augment', '--batch', '16', '--epochs', '25', '--seed', '0', '--workers', '5', '--val-workers', '2')
Step "train_sync_hr_vol_s0" { .\py.cmd scripts\train.py --run sync_hr_vol_T32_s0 @common --temporal-kw 'clock=hr,vol_readout=True,aux_weight=1.0' }
Step "train_sync_none_vol_s0" { .\py.cmd scripts\train.py --run sync_none_vol_T32_s0 @common --temporal-kw 'clock=none,vol_readout=True,aux_weight=1.0' }
Step "train_sync_hr_s0" { .\py.cmd scripts\train.py --run sync_hr_T32_s0 @common --temporal-kw 'clock=hr' }
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ALL DONE" | Tee-Object -Append "$logs\run_sync_pilots.log"
