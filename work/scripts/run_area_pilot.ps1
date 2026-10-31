# Pilot D: sync block with the weakly supervised area readout (soft LV mask on the layer3 map), seed 0,
# judged on VAL only. Then the baseline seeds (run_baseline_seeds.ps1).
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..
$logs = "E:\echo_project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
function Step($name, [scriptblock]$cmd) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $name" | Tee-Object -Append "$logs\run_area_pilot.log"
    & $cmd 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 "$logs\$name.log"
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] END $name exit=$LASTEXITCODE" | Tee-Object -Append "$logs\run_area_pilot.log"
}
$common = @('--temporal', 'sync', '--n-layers', '2', '--T', '32', '--stride', '2', '--pad', 'loop', '--pool-valid',
            '--augment', '--batch', '16', '--epochs', '25', '--seed', '0', '--workers', '5', '--val-workers', '2')
Step "train_sync_hr_area_s0" { .\py.cmd scripts\train.py --run sync_hr_area_T32_s0 @common --temporal-kw 'clock=hr,vol_readout=True,area_readout=True,aux_weight=1.0' }
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] AREA PILOT DONE" | Tee-Object -Append "$logs\run_area_pilot.log"
& "$PSScriptRoot\run_baseline_seeds.ps1"
