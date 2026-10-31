# Recipe pilots (seed 0, VAL only): does a stronger shared training recipe reduce the overfitting that
# limits every model (BiGRU train MSE 0.06 vs val 0.25)? Recipe "strong" = strong augmentation
# (rotation/scale, intensity, time reversal) + weight decay 1e-2; everything else as before.
#   E: BiGRU, strong recipe            (vs bigru_T32_s0, basic recipe)
#   F: sync + area readout + boundary fixes, strong recipe   (vs E)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..
$logs = "E:\echo_project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
function Step($name, [scriptblock]$cmd) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $name" | Tee-Object -Append "$logs\run_recipe_pilots.log"
    & $cmd 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 "$logs\$name.log"
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] END $name exit=$LASTEXITCODE" | Tee-Object -Append "$logs\run_recipe_pilots.log"
}
$common = @('--T', '32', '--stride', '2', '--pad', 'loop', '--pool-valid', '--augment', '--aug', 'strong', '--wd', '1e-2',
            '--batch', '16', '--epochs', '25', '--seed', '0', '--workers', '6', '--val-workers', '2')
Step "train_bigru_strong_s0" { .\py.cmd scripts\train.py --run bigru_strong_T32_s0 --temporal bigru --n-layers 3 @common }
Step "train_sync_area_strong_s0" { .\py.cmd scripts\train.py --run sync_area_strong_T32_s0 --temporal sync --n-layers 2 @common --temporal-kw 'clock=hr,vol_readout=True,area_readout=True,aux_weight=1.0,conv_pad=replicate,vol_trim=2' }
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ALL DONE" | Tee-Object -Append "$logs\run_recipe_pilots.log"
