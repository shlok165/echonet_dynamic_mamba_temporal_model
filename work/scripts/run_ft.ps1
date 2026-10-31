# Fine-tune the PanEcho encoder (last -ntune ConvNeXt blocks) with a temporal block, warm-started from the
# matching frozen-feature run (pe3_<name>_<init>_s<seed>), then evaluate on VAL at T 32. Identical protocol
# for every block (T 32, stride 2, loop pad, pool-valid, batch 16, AdamW, cosine, EMA). No test-set evaluation.
# Usage: powershell -File scripts\run_ft.ps1 [-only "bigru,sync_novol"] [-seeds "0"] [-epochs 5] [-lr 1e-4]
#        [-enclr 3e-5] [-ntune 3] [-ema 0.999] [-init ema] [-tag ft] [-log run_ft]
param([string]$only = "bigru,sync_novol", [string]$seeds = "0", [int]$epochs = 5, [string]$lr = "1e-4",
      [string]$enclr = "3e-5", [int]$ntune = 3, [string]$ema = "0.999", [string]$init = "ema",
      [string]$tag = "ft", [string]$log = "run_ft", [switch]$noeval)
$seedList = @($seeds -split ',' | ForEach-Object { [int]$_ })
$onlyList = @($only -split ',')
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..
$logs = "E:\echo_project\logs"
function Step($name, [scriptblock]$cmd) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $name" | Tee-Object -Append "$logs\$log.log"
    & $cmd 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 "$logs\$name.log"
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] END $name exit=$LASTEXITCODE" | Tee-Object -Append "$logs\$log.log"
}
# same blocks as run_feat_seeds.ps1
$configs = @{
    'bigru'            = @('bigru', 3, '');
    'mamba'            = @('mamba', 2, 'expand=1');
    'sync_novol'       = @('sync', 2, 'clock=hr,conv_pad=replicate');
    'sync_novol_fixed' = @('sync', 2, 'clock=fixed,conv_pad=replicate');
    'sync_novol_noclk' = @('sync', 2, 'clock=none,conv_pad=replicate')
}
$common = @('--panecho-tune', "$ntune", '--T', '32', '--stride', '2', '--pad', 'loop', '--pool-valid', '--batch', '16',
            '--epochs', "$epochs", '--lr', $lr, '--enc-lr', $enclr, '--ema', $ema, '--workers', '4', '--val-workers', '2')
foreach ($seed in $seedList) {
    foreach ($name in $onlyList) {
        $c = $configs[$name]
        $run = "ft_${name}_${tag}_s$seed"
        $src = "pe3_${name}_${init}_s$seed"
        $kw = @(); if ($c[2]) { $kw = @('--temporal-kw', $c[2]) }
        Step "train_$run" { .\py.cmd scripts\train.py --run $run --temporal $c[0] --n-layers $c[1] --init-from $src @common --seed $seed @kw }
        if (-not $noeval) { Step "evalVAL_$run" { .\py.cmd scripts\evaluate.py --run $run --split VAL --Ts 32 --workers 4 } }
    }
}
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ALL DONE" | Tee-Object -Append "$logs\$log.log"
