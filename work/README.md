# Project 3 — Mamba-based temporal modelling of cardiac motion (EchoNet-Dynamic)

All code, caches, checkpoints and results live here. `..\EchoNet-Dynamic` is read-only input.

```
work/
  py.cmd                 launcher: sets PATH like `conda activate` (needed for cuDNN + MKL DLLs)
  cardiac/               importable package
    config.py            paths, EF standardisation constants
    dataset.py           EchoClips (official split, on-the-fly cv2 decoding, clip sampling,
                         frame shuffle/reverse), EpochSampler, GPU Normalize, pixel stats
    models.py            ResNet-18 encoder; temporal blocks: identity / bigru / mamba (pure-PyTorch
                         selective scan, parallel + sequential); EFModel
    metrics.py           EF regression, Bland-Altman, AUROC, temporal smoothness, EF std across clips,
                         PCA, periodicity, ED/ES ridge probe + windowed localisation
  scripts/
    make_ed_es.py        derive + sanity-check ED/ES frames -> cache/ed_es_frames.csv
    benchmark.py         step time / peak VRAM / max batch vs T for each temporal block
    train.py             resumable training (identical protocol, only --temporal differs)
  tests/                 unittest suites (metrics, dataset, models)
  cache/                 ed_es_frames.csv, pixel_stats.json, torch hub weights
  runs/<run>/            args.json, info.json, log.jsonl, last.pt, best.pt
  results/               benchmark.json, evaluation outputs, figures
```

## UVT (Reynaud et al., MICCAI 2021) — external reference

Re-implemented from https://github.com/HReynaud/UVT (MIT licence) in `cardiac/uvt.py` (model, loss,
ED/ES peak decoding, loader for released weights) and `cardiac/uvt_data.py` (their 'repeat'/'sample'/'full'
preprocessing). Scripts: `scripts/train_uvt.py`, `scripts/eval_uvt.py`, `scripts/download_uvt_weights.py`.
Parameter names match the original so the released checkpoints load directly.

* Paper size (emb 1024, 16 BERT layers, FFN 8192) is ~336M parameters: inference fits the 6 GB GPU,
  training does not (`--preset small` trains a reduced version).
* The original adds no positional information to the Transformer (it uses `BertModel(...).encoder`
  directly), so it is order-blind; `--pos-emb` adds learned positions.
* UVT is trained with ED/ES supervision, so it is an external reference, not a protocol-matched model.
  The protocol-matched version is `--temporal uvt` in `scripts/train.py` (UVT-style Transformer block,
  learned positions, ~1.06M parameters, EF loss only), i.e. the brief's optional temporal-Transformer
  baseline.
* The original loader takes the first-listed traced frame as ES; in this dataset copy that frame is ED in
  10,006 / 10,024 videos, so the verified table is the default (`--label-order csv` reproduces the original).

## Running

Always go through `py.cmd` (from PowerShell, in this folder):

```
.\py.cmd -m unittest discover -s tests -v
.\py.cmd scripts\make_ed_es.py
.\py.cmd scripts\benchmark.py
.\py.cmd scripts\train.py --run <name> --temporal bigru ...
```

## Environment notes (found while setting up)

* Calling `...\envs\python\python.exe` directly (no `conda activate`) leaves `Library\bin` off PATH.
  Symptoms: `Cannot load symbol cudnnGetVersion` (process exits on the first conv) and a hard crash
  `0xC06D007F` in numpy on any multithreaded MKL call (e.g. `x @ x` for 20k floats; small arrays work).
  `py.cmd` fixes both; `cardiac/__init__.py` applies the same PATH fix if imported before torch/numpy.
* Windows drivers can silently spill VRAM into shared RAM. Scripts cap the allocator with
  `torch.cuda.set_per_process_memory_fraction` so over-budget runs fail with OOM instead.
* No sklearn/pytest in the env; metrics use numpy/scipy, tests use `unittest`.

## Data facts verified here (scripts/make_ed_es.py)

* Six FileList videos (5 TRAIN, 1 TEST) have no tracings; they are the 768x1024 header-quirk videos.
  One traced video (`0X4F8859C8AB4DA9CB`) is not in FileList -> 10,024 usable traced videos.
* Traced frames: 19,942 have 21 rows; 105 have 2-5 stacked 21-row tracings (none identical; volumes
  are averaged over blocks); 3 have 16/18/41 rows.
* ED = larger method-of-disks volume. Check: EF from traced volumes vs FileList EF, Pearson 0.967,
  MAE 1.07 EF points (a swapped assignment would give negative EF). ED precedes ES in 9,928 videos.
