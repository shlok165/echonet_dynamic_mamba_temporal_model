"""Central paths. The dataset folder is treated as read-only.

Code lives in WORK (on C:); everything we write (caches, checkpoints, results) goes under DATA_OUT, which
defaults to E:\\echo_project because C: is short on space. Override with the ECHO_DATA environment variable.
"""
import os
from pathlib import Path

PROJECT = Path(r"C:\Users\SHLOK\Desktop\AI\project")
DATA_ROOT = PROJECT / "EchoNet-Dynamic"
VIDEO_DIR = DATA_ROOT / "Videos"
FILELIST = DATA_ROOT / "FileList.csv"
TRACINGS = DATA_ROOT / "VolumeTracings.csv"

WORK = PROJECT / "work"
DATA_OUT = Path(os.environ.get("ECHO_DATA", r"E:\echo_project"))
CACHE = DATA_OUT / "cache"
RUNS = DATA_OUT / "runs"
RESULTS = DATA_OUT / "results"

ED_ES_CSV = CACHE / "ed_es_frames.csv"
STATS_JSON = CACHE / "pixel_stats.json"
TORCH_HOME = DATA_OUT / "torch-cache"

# Dataset-wide EF statistics (TRAIN split) used to standardise the regression target.
EF_MEAN = 55.75
EF_STD = 12.3
