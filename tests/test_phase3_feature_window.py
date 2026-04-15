from __future__ import annotations

import unittest
from unittest.mock import patch

from app.gestures.disambiguate import Decision
from app.gestures.engine import GestureEngine
from app.gestures.features import FeatureVector, HandInputQuality
from app.gestures.registry import GestureSnapshot
from app.gestures.temporal import FeatureTemporalCfg, FeatureWindow


class FeatureWindowTests(unittest.TestCase):
    def test_feature_window_uses_rolling_median(self):
        window = FeatureWindow(FeatureTemporalCfg(window=5, min_frames=3))

        window.update(FeatureVector(values=(1.0, 10.0), schema_version="phase3.v2"))
        window.update(FeatureVector(values=(2.0, 100.0), schema_version="phase3.v2"))
        out = window.update(FeatureVector(values=(50.0, 11.0), schema_version="phase3.v2"))

        self.assertTrue(out.ready)
        self.assertEqual(out.window_size, 3)
        self.assertIsNotNone(out.smoothed)
        self.assertEqual(out.smoothed.values, (2.0, 11.0))

    def test_feature_window_caps_to_configured_window_length(self):
        window = FeatureWindow(FeatureTemporalCfg(window=3, min_frames=1))

        window.update(FeatureVector(values=(1.0,), schema_version="phase3.v2"))
        window.update(FeatureVector(values=(2.0,), schema_version="phase3.v2"))
        window.update(FeatureVector(values=(3.0,), schema_version="phase3.v2"))
        out = window.update(FeatureVector(values=(100.0,), schema_version="phase3.v2"))

        self.assertEqual(out.window_size, 3)
        self.assertEqual(out.smoothed.values, (3.0,))

    def test_feature_window_resets_on_missing_frame(self):
        window = FeatureWindow(FeatureTemporalCfg(window=5, min_frames=3))
        window.update(FeatureVector(values=(1.0,), schema_version="phase3.v2"))
        cleared = window.update(None)

        self.assertFalse(cleared.ready)
        self.assertEqual(cleared.window_size, 0)
        self.assertIsNone(cleared.smoothed)

    def test_feature_window_rejects_schema_mismatch(self):
        window = FeatureWindow(FeatureTemporalCfg(window=5, min_frames=1))
        window.update(FeatureVector(values=(1.0, 2.0), schema_version="phase3.v2"))

        with self.assertRaises(ValueError):
            window.update(FeatureVector(values=(1.0,), schema_version="phase3.v2"))

        window.reset()
        window.update(FeatureVector(values=(1.0, 2.0), schema_version="phase3.v2"))

        with self.assertRaises(ValueError):
            window.update(FeatureVector(values=(1.0, 2.0), schema_version="phase3.v3"))

    def test_feature_window_flags_instability(self):
        window = FeatureWindow(
            FeatureTemporalCfg(window=5, min_frames=3, instability_threshold=0.05)
        )
        window.update(FeatureVector(values=(1.0, 1.0), schema_version="phase3.v2"))
        window.update(FeatureVector(values=(1.01, 0.99), schema_version="phase3.v2"))
        unstable = window.update(FeatureVector(values=(3.0, -2.0), schema_version="phase3.v2"))

        self.assertTrue(unstable.ready)
        self.assertFalse(unstable.passed)
        self.assertEqual(unstable.reason, "unstable")
        self.assertGreater(unstable.instability_score, 0.05)

    def test_feature_window_accepts_low_jitter(self):
        window = FeatureWindow(
            FeatureTemporalCfg(window=5, min_frames=3, instability_threshold=0.05)
        )
        window.update(FeatureVector(values=(1.0, 1.0), schema_version="phase3.v2"))
        window.update(FeatureVector(values=(1.01, 0.99), schema_version="phase3.v2"))
        stable = window.update(FeatureVector(values=(1.02, 1.01), schema_version="phase3.v2"))

        self.assertTrue(stable.ready)
        self.assertTrue(stable.passed)
        self.assertEqual(stable.reason, "ok")
        self.assertLessEqual(stable.instability_score, 0.05)

    def test_engine_keeps_detecting_when_feature_window_is_unstable(self):
        engine = GestureEngine(
            feature_temporal_cfg=FeatureTemporalCfg(window=3, min_frames=3, instability_threshold=0.05)
        )
        quality = HandInputQuality(True, "ok", 1.0, 1.0, 1.0, 1.0)
        vectors = [
            FeatureVector(values=(1.0, 1.0), schema_version="phase3.v2"),
            FeatureVector(values=(1.01, 0.99), schema_version="phase3.v2"),
            FeatureVector(values=(3.0, -2.0), schema_version="phase3.v2"),
        ]
        snapshot = GestureSnapshot(
            pinch=None,
            fist=False,
            closed_palm=False,
            open_palm=False,
            shaka=False,
            peace_sign=False,
            number=None,
            l_gesture=False,
            bravo=False,
            thumbs_down=False,
            point_right=False,
            point_left=False,
        )

        with patch("app.gestures.engine.assess_hand_input_quality", return_value=quality):
            with patch("app.gestures.engine.extract_feature_vector", side_effect=vectors):
                with patch.object(engine.registry, "detect", return_value=snapshot) as detect_mock:
                    with patch("app.gestures.engine.snapshot_to_candidates", return_value=set()):
                        with patch(
                            "app.gestures.engine.choose_one",
                            return_value=Decision(active=None, candidates=set(), reason="none"),
                        ):
                            engine.process(object())
                            engine.process(object())
                            out = engine.process(object())

        self.assertEqual(out.decision.reason, "none")
        self.assertTrue(out.feature_temporal.ready)
        self.assertFalse(out.feature_temporal.passed)
        self.assertIsNotNone(out.snapshot)
        self.assertEqual(detect_mock.call_count, 3)


if __name__ == "__main__":
    unittest.main()
