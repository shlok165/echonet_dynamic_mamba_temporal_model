"""Tests for clip index logic (pure) and real-video loading (skipped if the dataset is absent)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiac  # noqa: E402,F401
from cardiac import config  # noqa: E402
from cardiac.dataset import EchoClips, clip_indices, clip_starts  # noqa: E402


class TestClipIndices(unittest.TestCase):
    def test_long_video(self):
        idx, v, s = clip_indices(171, 32, 2, start=10)
        np.testing.assert_array_equal(idx, 10 + 2 * np.arange(32))
        self.assertTrue(v.all())
        self.assertEqual(s, 2)
        with self.assertRaises(ValueError):
            clip_indices(171, 32, 2, start=110)  # 110 + 62 > 170

    def test_loop(self):
        idx, v, _ = clip_indices(50, 32, 2, pad="loop")  # span 63 > 50
        self.assertEqual(idx.max(), 49 - 1)  # even positions only: 0..48
        np.testing.assert_array_equal(idx[:25], 2 * np.arange(25))
        self.assertEqual(idx[25], 0)  # 50 % 50
        self.assertEqual(v.sum(), 25)
        self.assertTrue(v[:25].all() and not v[25:].any())

    def test_repeat(self):
        idx, v, _ = clip_indices(50, 32, 2, pad="repeat")
        self.assertTrue((idx[25:] == 49).all())
        self.assertEqual(v.sum(), 25)

    def test_fit(self):
        idx, v, s = clip_indices(100, 64, 2, pad="fit")  # stride 2 span 127 > 100 -> stride 1
        self.assertEqual(s, 1)
        self.assertTrue(v.all())
        np.testing.assert_array_equal(idx, np.arange(64))
        idx, v, s = clip_indices(40, 64, 2, pad="fit")  # even stride 1 too long -> loop
        self.assertEqual(s, 1)
        self.assertEqual(v.sum(), 40)
        self.assertEqual(idx[40], 0)

    def test_starts(self):
        st = clip_starts(171, 32, 2, 4)
        self.assertEqual(st[0], 0)
        self.assertEqual(st[-1], 171 - 63)
        self.assertEqual(len(st), 4)
        self.assertEqual(clip_starts(50, 32, 2, 3), [0, 0, 0])


@unittest.skipUnless(config.VIDEO_DIR.exists(), "dataset not present")
class TestDataset(unittest.TestCase):
    def test_split_sizes(self):
        n = {s: len(EchoClips(s)) for s in ("TRAIN", "VAL", "TEST")}
        self.assertEqual(n, {"TRAIN": 7465, "VAL": 1288, "TEST": 1277})
        self.assertEqual(len(EchoClips("TEST", mode="multi", n_clips=3)), 3 * 1277)
        self.assertEqual(len(EchoClips("TEST", require_tracings=True)), 1276)

    def test_item(self):
        ds = EchoClips("TEST", T=16, stride=2, mode="multi", n_clips=2, max_videos=3)
        it = ds[1]
        self.assertEqual(tuple(it["video"].shape), (16, 3, 112, 112))
        self.assertEqual(str(it["video"].dtype), "torch.uint8")
        self.assertEqual(int(it["clip"]), 1)
        self.assertTrue(it["valid"].all())
        self.assertGreaterEqual(int(it["ed"]), 0)

    def test_shuffle_and_reverse_are_permutations(self):
        kw = dict(T=16, stride=2, mode="first", max_videos=2)
        a = EchoClips("TEST", **kw)[0]
        b = EchoClips("TEST", frame_order="shuffle", **kw)[0]
        b2 = EchoClips("TEST", frame_order="shuffle", **kw)[0]
        r = EchoClips("TEST", frame_order="reverse", **kw)[0]
        self.assertEqual(sorted(a["frame_idx"].tolist()), sorted(b["frame_idx"].tolist()))
        self.assertNotEqual(a["frame_idx"].tolist(), b["frame_idx"].tolist())
        self.assertEqual(b["frame_idx"].tolist(), b2["frame_idx"].tolist())  # deterministic
        self.assertEqual(r["frame_idx"].tolist(), a["frame_idx"].tolist()[::-1])
        # pixels follow the indices
        perm = [a["frame_idx"].tolist().index(i) for i in b["frame_idx"].tolist()]
        self.assertTrue((b["video"] == a["video"][perm]).all())

    def test_quirk_video_is_112(self):
        ds = EchoClips("TEST", T=8, stride=2, mode="first")
        v = int(np.where(ds.names == "0X5DD5283AC43CCDD1")[0][0])
        it = ds[v]
        self.assertEqual(tuple(it["video"].shape[-2:]), (112, 112))
        self.assertEqual(int(it["ed"]), -1)

    def test_rgb_order(self):
        import cv2
        ds = EchoClips("TEST", T=4, stride=1, mode="first", max_videos=1)
        x = ds[0]["video"][0].permute(1, 2, 0).numpy()
        cap = cv2.VideoCapture(str(config.VIDEO_DIR / f"{ds.names[0]}.avi"))
        _, bgr = cap.read()
        cap.release()
        np.testing.assert_array_equal(x, bgr[..., ::-1])


if __name__ == "__main__":
    unittest.main()
