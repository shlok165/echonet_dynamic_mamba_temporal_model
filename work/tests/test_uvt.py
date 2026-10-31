"""Tests for the UVT re-implementation (cardiac/uvt.py, cardiac/uvt_data.py, models.UVTTemporal)."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiac  # noqa: E402,F401
from cardiac import config  # noqa: E402
from cardiac.models import make_temporal  # noqa: E402
from cardiac.uvt import BertLayer, decode_sd_peaks, get_uvt, peak_errors, uvt_loss  # noqa: E402

TINY = dict(emb_dim=64, num_hidden_layers=2, intermediate_size=128, attention_heads=4)


class TestUVTModel(unittest.TestCase):
    def test_shapes_reg_and_cla(self):
        x = torch.rand(2, 6, 3, 128, 128)
        for mode, sd_shape in (("reg", (2, 6)), ("cla", (2, 6, 3))):
            m = get_uvt(SDmode=mode, **TINY).eval()
            with torch.no_grad():
                out = m(x)
            self.assertEqual(tuple(out["sd"].shape), sd_shape)
            self.assertEqual(tuple(out["ef"].shape), (2,))
            self.assertTrue(((out["ef"] > 0) & (out["ef"] < 1)).all())
            if mode == "reg":
                self.assertTrue(out["sd"].abs().max() <= 1)

    def test_state_dict_names_match_original(self):
        """Key names must equal those of the UVT repo (HF BertEncoder + ResNetAE) for released weights."""
        keys = set(get_uvt(**TINY).state_dict())
        for k in ("AE.encoder.input_conv.0.weight", "AE.encoder.res_blk_list.0.0.residual_block.0.weight",
                  "AE.encoder.multi_res_skip_list.3.1.running_mean", "AE.encoder.output_conv.1.weight", "AE.fc1.weight",
                  "T.layer.0.attention.self.query.weight", "T.layer.0.attention.self.key.bias",
                  "T.layer.0.attention.output.dense.weight", "T.layer.0.attention.output.LayerNorm.weight",
                  "T.layer.1.intermediate.dense.weight", "T.layer.1.output.dense.bias",
                  "T.layer.1.output.LayerNorm.bias", "extremas.0.weight", "extremas.6.weight", "ejection.3.weight"):
            self.assertIn(k, keys)
        self.assertFalse(any(k.startswith(("AE.decoder", "AE.fc2", "pos")) for k in keys))

    def test_ae_fc1_size(self):
        m = get_uvt(**TINY)
        self.assertEqual(tuple(m.AE.fc1.weight.shape), (64, 128 * 8 * 8))  # 128x128 input -> 8x8x128

    def test_bert_layer_matches_manual_postln(self):
        torch.manual_seed(0)
        l = BertLayer(16, 2, 32).eval()
        x = torch.randn(3, 5, 16)
        a = l.attention.self
        q, k, v = (m(x).view(3, 5, 2, 8).transpose(1, 2) for m in (a.query, a.key, a.value))
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(8), -1) @ v
        att = att.transpose(1, 2).reshape(3, 5, 16)
        h = l.attention.output.LayerNorm(l.attention.output.dense(att) + x)
        ref = l.output.LayerNorm(l.output.dense(torch.nn.functional.gelu(l.intermediate.dense(h))) + h)
        torch.testing.assert_close(l(x), ref, rtol=1e-5, atol=1e-5)

    def test_original_is_order_blind_and_pos_emb_fixes_it(self):
        torch.manual_seed(0)
        x = torch.rand(1, 6, 3, 128, 128)
        perm = torch.tensor([3, 0, 5, 1, 4, 2])
        m = get_uvt(**TINY).eval()  # as released: no positional information
        with torch.no_grad():
            a, b = m(x), m(x[:, perm])
        torch.testing.assert_close(a["sd"][:, perm], b["sd"], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(a["ef"], b["ef"], rtol=1e-4, atol=1e-6)  # EF invariant to shuffling
        mp = get_uvt(pos_emb=True, **TINY).eval()
        with torch.no_grad():
            a, b = mp(x), mp(x[:, perm])
        self.assertFalse(torch.allclose(a["sd"][:, perm], b["sd"], atol=1e-5))

    def test_padding_mask_isolates_padded_frames(self):
        torch.manual_seed(0)
        m = get_uvt(**TINY).eval()
        x = torch.rand(1, 6, 3, 128, 128)
        x2 = x.clone()
        x2[:, 4:] = torch.rand(1, 2, 3, 128, 128)  # change only padded frames
        mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        with torch.no_grad():
            a, b = m(x, mask), m(x2, mask)
        torch.testing.assert_close(a["sd"][:, :4], b["sd"][:, :4], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(a["ef"], b["ef"], rtol=1e-4, atol=1e-6)

    def test_loss(self):
        out = {"sd": torch.zeros(2, 4), "ef": torch.full((2,), 0.5)}
        y = torch.tensor([[1.0, 0, 0, -1], [0, 1, -1, 0]])
        loss, lsd, lef = uvt_loss(out, y, torch.tensor([50.0, 70.0]))
        self.assertAlmostEqual(lsd.item(), 0.5)  # mean of y^2
        self.assertAlmostEqual(lef.item(), 5 * (0 + 0.2 ** 2) / 2, places=6)
        att = torch.tensor([[1.0, 0, 0, 0], [0, 0, 0, 0]])
        _, lsd, _ = uvt_loss(out, y, torch.tensor([50.0, 70.0]), attention=att)
        self.assertAlmostEqual(lsd.item(), 1.0)


class TestDecoding(unittest.TestCase):
    def test_sinusoid_peaks(self):
        t = np.arange(150)
        sig = np.cos(2 * np.pi * (t - 20) / 50)  # ED (+1) at 20, 70, 120; ES (-1) at 45, 95, 145
        peaks = decode_sd_peaks(sig)
        ed = [i for c, i in peaks if c == 1]
        es = [i for c, i in peaks if c == -1]
        self.assertTrue(set(ed) >= {70, 120} and set(es) >= {45, 95})
        es_err, ed_err = peak_errors(peaks, es_frame=96, ed_frame=69)
        self.assertEqual((es_err, ed_err), (-1.0, 1.0))

    def test_too_few_crossings(self):
        self.assertIsNone(decode_sd_peaks(np.linspace(-1, 1, 40)))
        self.assertTrue(all(math.isnan(e) for e in peak_errors(None, 3, 30)))


class TestUVTTemporalBlock(unittest.TestCase):
    def test_shape_and_order_sensitivity(self):
        torch.manual_seed(0)
        blk = make_temporal("uvt", d_in=32, d_model=16, n_layers=2, heads=4, intermediate=24).eval()
        f = torch.randn(2, 10, 32)
        z = blk(f)
        self.assertEqual(tuple(z.shape), (2, 10, 16))
        perm = torch.randperm(10)
        self.assertFalse(torch.allclose(blk(f[:, perm]), z[:, perm], atol=1e-5))


@unittest.skipUnless(config.VIDEO_DIR.exists() and config.ED_ES_CSV.exists(), "dataset/ed_es table not present")
class TestUVTData(unittest.TestCase):
    def test_repeat_mode(self):
        from cardiac.uvt_data import UVTEchoSet
        ds = UVTEchoSet("TEST", mode="repeat", max_videos=3)
        it = ds[0]
        self.assertEqual(tuple(it["video"].shape), (128, 3, 128, 128))
        y = it["sd_label"].numpy()
        self.assertLessEqual(y.max(), 1.0)
        self.assertGreaterEqual(y.min(), -1.0)
        self.assertTrue((y == 1).any() or (y == -1).any())
        self.assertTrue(float(it["video"][:, :, :8].abs().max()) == 0.0)  # zero padding rows

    def test_full_mode_labels(self):
        from cardiac.uvt_data import UVTEchoSet
        ds = UVTEchoSet("TEST", mode="full", max_videos=2)
        it = ds[0]
        y = it["sd_label"].numpy()
        self.assertEqual(int(np.where(y == -1)[0][0]), int(it["es"]))
        self.assertEqual(int(np.where(y == 1)[0][0]), int(it["ed"]))
        self.assertEqual(it["video"].shape[0], len(y))

    def test_sample_mode_lengths(self):
        from cardiac.uvt_data import UVTEchoSet
        ds = UVTEchoSet("TEST", mode="sample", max_videos=20)
        for i in range(len(ds)):
            it = ds[i]
            self.assertEqual(it["video"].shape[0], 128)
            self.assertEqual(it["sd_label"].shape[0], 128)
            self.assertGreater(it["attention"].sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
