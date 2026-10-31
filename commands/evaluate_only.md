# Evaluate trained models

Run in PowerShell from `C:\Users\SHLOK\Desktop\AI\project\work`. `--split VAL` = validation; remove it for the test set.
`_s0` = seed 0 (`_s1`, `_s2` for the other seeds).

| Model | Command |
|---|---|
| sync, fixed frequency (final) | `.\py.cmd scripts\evaluate.py --run pe3_sync_novol_fixed_ema_s0 --split VAL` |
| sync, heart-rate clock | `.\py.cmd scripts\evaluate.py --run pe3_sync_novol_ema_s0 --split VAL` |
| sync, tempered clock | `.\py.cmd scripts\evaluate.py --run pe3_sync_novol_tempered_ema_s0 --split VAL` |
| sync, learned angles | `.\py.cmd scripts\evaluate.py --run pe3_sync_novol_learned_ema_s0 --split VAL` |
| sync, no rotation | `.\py.cmd scripts\evaluate.py --run pe3_sync_novol_noclk_ema_s0 --split VAL` |
| sync + volumetric readout | `.\py.cmd scripts\evaluate.py --run pe3_sync_s0 --split VAL` |
| BiGRU | `.\py.cmd scripts\evaluate.py --run pe3_bigru_ema_s0 --split VAL` |
| Mamba | `.\py.cmd scripts\evaluate.py --run pe3_mamba_ema_s0 --split VAL` |
| Transformer (UVT) | `.\py.cmd scripts\evaluate.py --run pe3_uvt_ema_s0 --split VAL` |
| No temporal model | `.\py.cmd scripts\evaluate.py --run pe3_identity_ema_s0 --split VAL` |
| PanEcho zero-shot | `.\py.cmd scripts\panecho_zeroshot.py --split VAL --packing quirk` |
| PanEcho ridge probe | `.\py.cmd scripts\panecho_probe.py --stage ridge` |
| PanEcho Transformer, fine-tuned, 16-frame clips | `.\py.cmd scripts\stats.py --split val --T 16 --models panecho_tune_lr0.0001_s0` |
| PanEcho Transformer, fine-tuned, our clips | `.\py.cmd scripts\stats.py --split val --T 32 --models panecho_tune_s3_lr0.0003_s0` |
| BiGRU, encoder fine-tuned | `.\py.cmd scripts\evaluate.py --run ft_bigru_ft_s0 --split VAL` |
| sync + harmonic + range readouts (single-frame features) | `.\py.cmd scripts\evaluate.py --run pe_sync_harm_range_s0 --split VAL` |
| BiGRU + harmonic + range readouts (single-frame features) | `.\py.cmd scripts\evaluate.py --run pe_bigru_harm_range_s0 --split VAL` |
| End-to-end ResNet-18 + BiGRU | `.\py.cmd scripts\evaluate.py --run bigru_T32_s0 --split VAL` |
| End-to-end ResNet-18 + Mamba | `.\py.cmd scripts\evaluate.py --run mamba_T32_s0 --split VAL` |
| End-to-end ResNet-18 + Transformer (UVT) | `.\py.cmd scripts\evaluate.py --run uvt_T32_s0 --split VAL` |
| End-to-end ResNet-18 + sync + latent volume | `.\py.cmd scripts\evaluate.py --run sync_hr_vol_T32_s0 --split VAL` |
| End-to-end ResNet-18 + sync + area readout | `.\py.cmd scripts\evaluate.py --run sync_hr_area_T32_s0 --split VAL` |
