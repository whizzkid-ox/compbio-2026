"""Fast orchestration tests; never launch a full-dataset simulation."""
import ast
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import numpy as np
import tables
import snn_full_common as full


class FullWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = full.prototype("AdEx")
        cls.base = full.full_config(cls.p, "AdEx")

    def test_eleven_unique_ofat_and_ff_reuse(self):
        m = full.build_manifest("AdEx", asdict(self.base), self.p.seeds_for(self.base))
        self.assertEqual(m["unique_configurations"], 11)
        self.assertEqual(len({r["config_id"] for r in m["configurations"]}), 11)
        self.assertEqual(set(m["disabled_parameters"]), {"fan_in", "width"})
        for r in m["configurations"]:
            self.assertEqual(r["condition_to_run"], "REC")
            self.assertEqual(r["FF_reference"], "baseline_FF/events.npz")
            cmd = full.sweep_command(m, r, Path("planned"))
            self.assertEqual(cmd[cmd.index("--conditions") + 1], "REC")
            self.assertNotIn("--smoke-test", cmd)
            if r["parameter_name"] == "rec_gain":
                self.assertAlmostEqual(r["rec_inh_g"] / r["rec_exc_g"], 4.)
            if r["parameter_name"] == "EI_ratio":
                self.assertEqual(r["rec_exc_g"], 3e-5)
        zero = next(r for r in m["configurations"] if r["config_id"] == "rec_probability_0p00")
        self.assertEqual(zero["rec_probability"], 0.)

    def test_optional_input_sweeps_and_duplicate_ff(self):
        m = full.build_manifest("AdEx", asdict(self.base), self.p.seeds_for(self.base), True, True)
        self.assertEqual(len(m["configurations"]), 17)
        for r in m["configurations"]:
            if r["parameter_name"] in {"fan_in", "width"}:
                self.assertEqual(r["condition_to_run"], "both")
                self.assertEqual(r["FF_reference"], r["output_directory"] + "/events.npz")
                # Connectivity generation only: no neural simulation.
                c = self.p.Config(**r["full_parameters"])
                inputs, rec, _ = self.p.connectivity(c, self.p.seeds_for(c))
                self.assertTrue(np.all(inputs.sum(0) == c.fan_in))
                base_input, base_rec, _ = self.p.connectivity(self.base, self.p.seeds_for(self.base))
                self.assertFalse(np.array_equal(inputs, base_input))
                np.testing.assert_array_equal(rec, base_rec)
        m = full.build_manifest("AdEx", asdict(self.base), self.p.seeds_for(self.base), rerun_ff=True)
        self.assertIsNone(m["FF_reference_job"])
        self.assertTrue(all(r["condition_to_run"] == "both" for r in m["configurations"]))

    def test_recurrent_parameters_leave_input_unchanged(self):
        m = full.build_manifest("AdEx", asdict(self.base), self.p.seeds_for(self.base))
        original, _, _ = self.p.connectivity(self.base, self.p.seeds_for(self.base))
        for r in m["configurations"]:
            c = self.p.Config(**r["full_parameters"])
            inputs, rec, _ = self.p.connectivity(c, self.p.seeds_for(c))
            np.testing.assert_array_equal(inputs, original)
            self.assertFalse(np.diag(rec).any())
            if c.rec_probability == 0:
                self.assertFalse(rec.any())

    def test_complete_english_selection_and_smoke_bounds(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "shd_train.h5"
            with tables.open_file(path, "w") as f:
                f.create_array("/", "labels", np.array([12, 9, 0, 5, 19, 0]))
                g = f.create_group("/", "extra")
                f.create_array(g, "speaker", np.arange(6) + 10)
            c = replace(self.base, data_path=str(path))
            ids, labels, speakers, english, total = full.select_full(self.p, c)
            np.testing.assert_array_equal(ids, [1, 2, 3, 5])
            np.testing.assert_array_equal(labels, [9, 0, 5, 0])
            np.testing.assert_array_equal(speakers, [11, 12, 13, 15])
            self.assertEqual((english, total), (4, 6))
            for ids in ("0", "1,1", "1,2,3"):
                with self.assertRaises(ValueError):
                    full.select_full(self.p, c, True, ids)
            with self.assertRaises(ValueError):
                full.select_full(self.p, c, False, "1")

    def test_checkpoints_failed_trials_orphans_and_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "checkpoints").mkdir()
            meta = {"model": "AdEx", "conditions": ["FF"], "trial_ids": [3, 8], "committed_trials": 0,
                    "chunks": [], "validation": {}, "status": "running"}
            shared = {"trial_ids": np.array([3, 8]), "labels": np.array([0, 1]), "speaker_ids": np.array([4, 5]),
                      "trial_order": np.arange(2), "source_split": np.array(["train", "train"]),
                      "neuron_ids": np.arange(2), "population": np.array(["E", "I"])}
            full.atomic_npz(out / "shared.npz", shared)
            empty = (np.empty(0), np.empty(0, dtype=np.int32))
            block = [{"id": 3, "hash": "a"*64, "events": {"FF": (np.array([1., 2.]), np.array([0, 1], dtype=np.int32)), "FF_pre": empty},
                      "valid": {"FF": True}, "rows": [{"trial_id": 3, "condition": "FF", "valid": True, "mean_rate_hz": 2., "error": ""}]}]
            full.commit_block(self.p, out, meta, block)
            self.assertEqual(meta["committed_trials"], 1)
            full.finalise(out, meta, self.p)
            with np.load(out / "events.npz", allow_pickle=False) as z:
                np.testing.assert_array_equal(z["trial_ids"], [3])
                np.testing.assert_array_equal(z["planned_trial_ids"], [3, 8])
            # Unjournalled chunk must never enter resumed data.
            (out / "checkpoints/orphan.npz").write_bytes(b"uncommitted")
            resumed = json.loads((out / "metadata.json").read_text())
            block = [{"id": 8, "hash": "b"*64, "events": {"FF": empty, "FF_pre": empty}, "valid": {"FF": False},
                      "rows": [{"trial_id": 8, "condition": "FF", "valid": False, "mean_rate_hz": None, "error": "Injected numerical failure"}]}]
            full.commit_block(self.p, out, resumed, block)
            full.finalise(out, resumed, self.p)
            self.assertEqual(resumed["status"], "partially_valid")
            self.assertTrue((out / "PARTIAL.txt").exists())
            self.assertFalse((out / "COMPLETE.txt").exists())
            with np.load(out / "events.npz", allow_pickle=False) as z:
                np.testing.assert_array_equal(z["FF_offsets"], [0, 2, 2])
                np.testing.assert_array_equal(z["FF_valid"], [True, False])
                np.testing.assert_array_equal(z["FF_times_ms"], [1., 2.])
                self.assertTrue(all(z[k].dtype != object for k in z.files))
            self.assertIsNone(resumed["failures"][0]["mean_rate_hz"])
            damaged = out / resumed["chunks"][0]["file"]
            with damaged.open("ab") as f:
                f.write(b"damage")
            with self.assertRaises(AssertionError):
                list(full.checked_chunks(out, resumed))

    def test_full_numerical_failure_and_programming_error_paths(self):
        # One synthetic trial through the real orchestration, without integration.
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "shd_train.h5"
            with tables.open_file(source, "w") as f:
                f.create_array("/", "labels", np.array([0]))
                group = f.create_group("/", "spikes")
                times = f.create_vlarray(group, "times", tables.Float64Atom())
                units = f.create_vlarray(group, "units", tables.Int64Atom())
                times.append(np.array([.005])); units.append(np.array([0]))
            def record(model, waveform, c, **kwargs):
                return (np.full((c.n_out, c.n_steps+1), -70.),
                        np.zeros((c.n_out, c.n_steps+1), dtype=bool),
                        np.zeros((c.n_out, c.n_steps+1)))
            calls = [0]
            def numerical(model, waveform, c, **kwargs):
                calls[0] += 1
                if calls[0] == 1:
                    raise FloatingPointError("INJECTED: not a scientific simulation")
                return record(model, waveform, c, **kwargs)
            args = ["simulate_ff_rec_adex_full.py", "--smoke-test", "--trial-ids", "0",
                    "--data-path", str(source), "--n-exc", "2", "--n-inh", "1", "--fan-in", "2",
                    "--settle-ms", "5", "--input-ms", "20", "--post-ms", "5", "--run-dir", str(Path(td)/"numerical")]
            with patch.object(sys, "argv", args), patch.object(self.p, "simulate", numerical):
                with self.assertRaises(SystemExit) as error:
                    full.full_main("AdEx", str(full.ROOT/"simulate_ff_rec_adex_full.py"))
                self.assertEqual(error.exception.code, 2)
            out = Path(td)/"numerical"
            m = json.loads((out/"metadata.json").read_text())
            self.assertEqual(m["status"], "partially_valid")
            self.assertEqual(len(m["failures"]), 1)
            self.assertIsNone(m["failures"][0]["mean_rate_hz"])
            with np.load(out/"events.npz", allow_pickle=False) as z:
                self.assertFalse(z["FF_valid"][0]); self.assertTrue(z["REC_valid"][0])
            with patch.object(sys, "argv", ["full", "--resume", str(out), "--trial-ids", "1"]):
                with self.assertRaisesRegex(ValueError, "cannot change trial selection"):
                    full.full_main("AdEx", str(full.ROOT/"simulate_ff_rec_adex_full.py"))
            args[-1] = str(Path(td)/"programming_error")
            with patch.object(sys, "argv", args), patch.object(self.p, "simulate", side_effect=RuntimeError("INJECTED API error")):
                with self.assertRaisesRegex(RuntimeError, "API error"):
                    full.full_main("AdEx", str(full.ROOT/"simulate_ff_rec_adex_full.py"))
            m = json.loads((Path(args[-1])/"metadata.json").read_text())
            self.assertEqual(m["status"], "error")
            self.assertEqual(m["committed_trials"], 0)
            self.assertTrue(list(Path(args[-1]).glob("ERROR_*.json")))

    def test_exclusive_run_lock(self):
        with tempfile.TemporaryDirectory() as td:
            with full.run_lock(td):
                with self.assertRaises(OSError):
                    with full.run_lock(td):
                        pass
            with full.run_lock(td):
                pass

    def test_prototype_rename_preserves_all_code(self):
        before = Path("results/full_implementation_validation/adex_before_rename.txt")
        if not before.exists():
            self.skipTest("Local pre-rename snapshot is not distributed to HPC")
        original = before.read_text().replace("simulate_ff_rec.py", "simulate_ff_rec_adex.py")
        self.assertEqual(ast.dump(ast.parse(original)), ast.dump(ast.parse(Path(self.p.__file__).read_text())))
        hh = Path("results/full_implementation_validation/hh_before.txt").read_bytes()
        self.assertEqual(hh, Path("simulate_ff_rec_hh.py").read_bytes())


if __name__ == "__main__":
    unittest.main()
