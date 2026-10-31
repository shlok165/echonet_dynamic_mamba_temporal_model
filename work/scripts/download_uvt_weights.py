r"""Download the released UVT checkpoints (links from download_weights.sh of https://github.com/HReynaud/UVT).

Usage: py.cmd scripts\download_uvt_weights.py M REG      (M = mirror/'repeat', R = random/'sample';
                                                        REG = regression SD branch, CLA = classification)
Saves to cache/uvt_weights/UVT_<M|R>_<REG|CLA>/best.pt and checks that the file is a torch checkpoint
(not an HTML error page) before keeping it.
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardiac import config  # noqa: E402
from cardiac.uvt import RELEASED_WEIGHTS  # noqa: E402


def main():
    if len(sys.argv) != 3 or (sys.argv[1], sys.argv[2]) not in RELEASED_WEIGHTS:
        print("usage: download_uvt_weights.py [M|R] [REG|CLA]")
        sys.exit(2)
    key = (sys.argv[1], sys.argv[2])
    out = config.CACHE / "uvt_weights" / f"UVT_{key[0]}_{key[1]}" / "best.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("already present:", out)
        return
    tmp = out.with_suffix(".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if have:
        headers["Range"] = f"bytes={have}-"  # resume a partial download (files are ~4 GB)
    req = urllib.request.Request(RELEASED_WEIGHTS[key], headers=headers)
    with urllib.request.urlopen(req) as r:
        resumed = have and r.status == 206
        if have and not resumed:
            print("server ignored the resume request; restarting from 0")
        with open(tmp, "ab" if resumed else "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    with open(tmp, "rb") as f:
        head = f.read(4)
    if head[:2] != b"PK" and head[:1] != b"\x80":  # torch zip checkpoint or legacy pickle
        tmp.unlink()
        raise SystemExit(f"download is not a torch checkpoint (starts with {head!r}); the link may have expired")
    tmp.replace(out)
    print("saved", out, out.stat().st_size // 2 ** 20, "MiB")


if __name__ == "__main__":
    main()
