# Frozen PanEcho features: temporal blocks (+ sync ablations) x seeds, then VAL evaluation of each run.
# Identical protocol for all (T 32, stride 2, loop pad, pool-valid, batch 16, AdamW, cosine, one shared lr
# chosen by run_feat_lr.ps1 on VAL). No test-set evaluation here.
# Usage: powershell -File scripts\run_feat_seeds.ps1 -lr 3e-4 [-features panecho|panecho_s3] [-epochs 40]
#        [-seeds 0,1,2] [-only pe_sync,pe_bigru] [-log run_feat_seeds] [-extra "--ema,0.999" -tag ema]
# -extra: training arguments added to every run (comma-separated tokens); -tag: run-name suffix
# (<prefix>_<name>_<tag>_s<seed>), so recipe variants never overwrite the reference runs.
param([string]$lr = "3e-4", [string]$features = "panecho", [int]$epochs = 40, [string]$seeds = "0,1,2",
      [string]$only = "", [string]$log = "run_feat_seeds", [switch]$noeval, [string]$extra = "",
      [string]$tag = "")
# -File passes "a,b" as one string: split here
$seedList = @($seeds -split ',' | ForEach-Object { [int]$_ })
$onlyList = @(if ($only) { $only -split ',' })
$extraList = @(if ($extra) { $extra -split ',' })
if ($extraList.Count -and -not $tag) { throw "-extra needs a -tag" }
$suffix = if ($tag) { "_$tag" } else { "" }
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..
$logs = "E:\echo_project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
function Step($name, [scriptblock]$cmd) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $name" | Tee-Object -Append "$logs\$log.log"
    & $cmd 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 "$logs\$name.log"
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] END $name exit=$LASTEXITCODE" | Tee-Object -Append "$logs\$log.log"
}
$prefix = @{ 'panecho' = 'pe'; 'panecho_s3' = 'pe3'; 'panecho_s3_g3' = 'pe3g' }[$features]
$vol = 'vol_readout=True,aux_weight=1.0,conv_pad=replicate,vol_trim=2'
# name -> (temporal, n_layers, temporal-kw); all ~1.07-1.12M temporal parameters except identity
$configs = [ordered]@{
    'identity'          = @('identity', 2, '');
    'bigru'             = @('bigru', 3, '');
    'mamba'             = @('mamba', 2, 'expand=1');
    'uvt'               = @('uvt', 2, '');
    'sync'              = @('sync', 2, "clock=hr,$vol");
    'sync_novol'        = @('sync', 2, 'clock=hr,conv_pad=replicate');
    'sync_novol_noclk'  = @('sync', 2, 'clock=none,conv_pad=replicate');
    'sync_novol_fixed'  = @('sync', 2, 'clock=fixed,conv_pad=replicate');
    'sync_novol_learned' = @('sync', 2, 'clock=learned,conv_pad=replicate');
    'sync_novol_tempered' = @('sync', 2, 'clock=tempered,conv_pad=replicate,min_clock_T=24');
    'sync_novol_fixed_grid' = @('sync', 2, 'clock=fixed,conv_pad=replicate,grid_tokens=9');
    'bigru_grid'        = @('bigru', 3, 'grid_tokens=9');
    'sync_novol_fixed_gridreg' = @('sync', 2, 'clock=fixed,conv_pad=replicate,grid_tokens=9,grid_dim=32,grid_drop=0.3');
    'sync_novol_mix'    = @('sync', 2, 'clock=mix,conv_pad=replicate');
    'sync_novol_fixed_phase' = @('sync', 2, 'clock=fixed,conv_pad=replicate,pool=phase');
    'sync_novol_mix_phase' = @('sync', 2, 'clock=mix,conv_pad=replicate,pool=phase');
    'bigru_phase'       = @('bigru', 3, 'pool=phase');
    'sync_noclock'      = @('sync', 2, "clock=none,$vol");
    'sync_fixed'        = @('sync', 2, "clock=fixed,$vol");
    'sync_learned'      = @('sync', 2, "clock=learned,$vol");
    'sync_range'        = @('sync', 2, 'clock=hr,conv_pad=replicate,range_k=16');
    'sync_range_noclk'  = @('sync', 2, 'clock=none,conv_pad=replicate,range_k=16');
    'sync_range_vol'    = @('sync', 2, "clock=hr,range_k=16,$vol");
    'sync_harm'         = @('sync', 2, 'clock=hr,conv_pad=replicate,harm_k=16');
    'sync_harm_range'   = @('sync', 2, 'clock=hr,conv_pad=replicate,harm_k=16,range_k=16');
    'sync_harm_prior'   = @('sync', 2, 'clock=hr,conv_pad=replicate,harm_k=16,harm_clock=prior');
    'sync_harm_noclk'   = @('sync', 2, 'clock=none,conv_pad=replicate,harm_k=16');
    'bigru_range'       = @('bigru', 3, 'range_k=16');
    'bigru_harm'        = @('bigru', 3, 'harm_k=16');
    'bigru_harm_range'  = @('bigru', 3, 'harm_k=16,range_k=16')
}
$common = @('--features', $features, '--T', '32', '--stride', '2', '--pad', 'loop', '--pool-valid', '--batch', '16',
            '--epochs', "$epochs", '--lr', $lr, '--workers', '2', '--val-workers', '1')
foreach ($seed in $seedList) {
    foreach ($name in $configs.Keys) {
        if ($onlyList.Count -and -not ($onlyList -contains $name)) { continue }
        $c = $configs[$name]
        $run = "${prefix}_${name}${suffix}_s$seed"
        $kw = @(); if ($c[2]) { $kw = @('--temporal-kw', $c[2]) }
        Step "train_$run" { .\py.cmd scripts\train.py --run $run --temporal $c[0] --n-layers $c[1] @common --seed $seed @kw @extraList }
        if (-not $noeval) { Step "evalVAL_$run" { .\py.cmd scripts\evaluate.py --run $run --split VAL --workers 2 } }
    }
}
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ALL DONE" | Tee-Object -Append "$logs\$log.log"
