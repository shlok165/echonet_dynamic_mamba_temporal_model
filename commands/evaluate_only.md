# Evaluate the final model

PowerShell, from any folder. `--split VAL` = validation; remove it for the test set. `_s0` = seed 0 (`_s1`, `_s2` for the other seeds).

| sync, fixed frequency (final) | `cd C:\Users\SHLOK\Desktop\AI\project\work; .\py.cmd scripts\evaluate.py --run pe3_sync_novol_fixed_ema_s0 --split VAL` |

| BiGRU (baseline) | `cd C:\Users\SHLOK\Desktop\AI\project\work; .\py.cmd scripts\evaluate.py --run pe3_bigru_ema_s0 --split VAL` |

