# Temporal models for ejection-fraction estimation on EchoNet-Dynamic

Ejection fraction (EF) from echocardiography videos with a frozen frame encoder and a lightweight temporal
model. Every temporal model (a BiGRU, a bidirectional Mamba, a Transformer, no temporal model, and the
proposed **sync** block) runs on the same per-frame features of the frozen PanEcho ConvNeXt-T image encoder,
with the same clip protocol, training recipe and evaluation. Only the temporal model changes.

**Sync block.** This is a bidirectional selective state-space model (Mamba-2-style, exact matrix form) in which
part of the state rotates at a frequency. In the final variant, that frequency is one learned period shared by
all clips. Variants take it from a label-free heart-rate estimate of each clip, a tempered mix of the two, or
input-dependent angles (Mamba-3 style), or use no rotation at all.

## Results

Lower MAE is better. MAE is in EF points, using 4 clips per video and the mean of 3 seeds.

| Model | Validation MAE | Test MAE |
|---|---|---|
| sync, fixed frequency (final) | **3.963** | 4.043 |
| bidirectional Mamba | 3.998 | **4.020** |
| BiGRU | 4.056 | 4.105 |

- **Validation:** sync beats BiGRU by 0.093 MAE (paired bootstrap p = 0.011).
- **Test:** the sync advantage over BiGRU is smaller (0.062) and not significant (p = 0.093). Mamba has the
  lowest test MAE and beats BiGRU (p = 0.032).
- **Other metrics:** sync has the best RMSE, R², Pearson r and CCC on both sets.
- **Full tables:** all 13 models, including PanEcho's own video model as zero-shot, ridge probe and
  fine-tuned versions, are in
  [`architecture/all_architectures_and_comparison.pdf`](architecture/all_architectures_and_comparison.pdf).
  The test columns of some baselines are marked "pending" until their test evaluation finishes.

All design choices were made on the validation set. The test set was evaluated once, on the locked set of
models.

## Repository layout

```
architecture/   one PDF per proposed or explored model (architecture + comparison with BiGRU), and
                all_architectures_and_comparison.pdf (all architectures + side-by-side tables)
baseline/       baselines.pdf: architectures of all baselines and their results in tables
commands/       evaluate_only.md: the command that evaluates each trained model
work/           code
  cardiac/      package: datasets, temporal models (sync.py = proposed block), PanEcho encoder, metrics
  scripts/      training, evaluation, statistics, PanEcho baselines, figures and PDFs
  tests/        unit tests
```

## Data and weights (not included)

- **EchoNet-Dynamic** is not redistributed here. Request access from Stanford AIMI and follow its research use
  agreement. Please acknowledge Stanford AIMI and cite the dataset (https://doi.org/10.71718/yqp5-y078):

  ```bibtex
  @misc{stanford_aimi_echonet_dynamic_2026,
    url = {https://stanford.redivis.com/datasets/66s1-2hsmzj5rn?v=1.0},
    author = {{Stanford AIMI}},
    title = {EchoNet-Dynamic},
    publisher = {{Redivis (DOI:10.71778/V2DW-7A53)}},
    year = 2026, month = apr
  }
  ```

- **PanEcho** weights (Holste et al., JAMA 2025; https://github.com/CarDS-Yale/PanEcho, CC BY-NC-SA 4.0) are
  not included. Download `panecho.pt` from the PanEcho release into `<ECHO_DATA>\third_party\panecho_weights\`.
  The code loads tensors only and does not run the repository's hub code.
- **UVT** (`work/cardiac/uvt.py`) is a re-implementation of https://github.com/HReynaud/UVT (MIT licence,
  © the UVT authors).

## Running

The code is written for Windows and PowerShell with one NVIDIA GPU; it was developed on a 6 GB RTX 3050.
`work/py.cmd` launches the project virtual environment. The data and output root defaults to
`E:\echo_project` and can be changed with the `ECHO_DATA` environment variable.

The dataset location is set in `work/cardiac/config.py`.

The command that evaluates each model is in [`commands/evaluate_only.md`](commands/evaluate_only.md). The
training options are documented at the top of `work/scripts/train.py`, `work/scripts/run_feat_seeds.ps1` and
`work/scripts/panecho_probe.py`.
