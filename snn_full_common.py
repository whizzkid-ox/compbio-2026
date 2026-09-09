"""Shared orchestration/storage for full SHD runs and OFAT manifests.

Scientific integration, replay and connectivity remain in the validated prototypes.
Checkpoint chunks are immutable; metadata.json is the atomic commit journal.
Only --resume may update an existing run. Failed trials require C_valid masks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent
PROTOTYPES = {"AdEx": "simulate_ff_rec_adex", "HH": "simulate_ff_rec_hh"}
SKIP_CONFIG = {"classes", "digits", "all_trials", "trials_per_digit", "smoke_test", "validate",
               "sweep_input_g", "input_g_values", "save_sweep_events", "posthoc_input_g",
               "compare_posthoc_dir"}
REC_ONLY = {"rec_probability", "rec_exc_g", "rec_inh_g"}
GRIDS = {"rec_probability": [0., .05, .10, .20, .30], "rec_gain": [.5, 1., 2., 4.],
         "EI_ratio": [1., 2., 4., 8.], "fan_in": [30, 60, 90, 120], "width": [15., 35., 70., 140.]}
BASE_VALUES = {"rec_probability": .10, "rec_gain": 1., "EI_ratio": 4., "fan_in": 60, "width": 35.}


def prototype(model):
    return importlib.import_module(PROTOTYPES[model])


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def atomic_npz(path, arrays):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    for a in arrays.values():
        assert not np.asarray(a).dtype.hasobject
    with temp.open("xb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    with np.load(temp, allow_pickle=False) as z:
        assert set(z.files) == set(arrays)
        for key in arrays:
            np.testing.assert_array_equal(z[key], arrays[key], err_msg=key)
    os.replace(temp, path)


@contextmanager
def run_lock(directory):
    """OS-released advisory lock: a killed process does not leave a stale lock."""
    f = (Path(directory) / "run.lock").open("a+b")
    try:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        f.close()


def full_config(p, model):
    return p.Config(input_g=3e-5 if model == "AdEx" else 3e-4,
                    classes="english", all_trials=True, diagnostic_trials=0)


def full_parser(p, model):
    parser = argparse.ArgumentParser(description=f"{model}: all English SHD trials; checkpointed, unbatched JAX simulation.")
    default = full_config(p, model)
    for field in fields(p.Config):
        if field.name in SKIP_CONFIG:
            continue
        value = getattr(default, field.name)
        kw = {"default": value}
        if isinstance(value, bool):
            kw["action"] = "store_true"
        else:
            kw["type"] = int if value is None else type(value)
        if field.name in ("conditions", "split", "wiring"):
            kw["choices"] = {"conditions": ["FF", "REC", "both"], "split": ["train", "test"], "wiring": ["tonotopic", "random"]}[field.name]
        parser.add_argument("--" + field.name.replace("_", "-"), **kw)
    parser.add_argument("--smoke-test", action="store_true", help="Only two deterministic diagnostic trials, never full data")
    parser.add_argument("--trial-ids", default="", help="One or two explicit source IDs; requires --smoke-test")
    parser.add_argument("--plan-only", action="store_true", help="Inspect selection without building or simulating networks")
    parser.add_argument("--run-dir", help="Exclusive output path, for manifest jobs")
    parser.add_argument("--resume", help="Resume an existing run; stored scientific configuration is restored")
    parser.add_argument("--checkpoint-trials", type=int, default=25)
    parser.add_argument("--stop-after", type=int, help="Checkpoint and stop after this many new trials in this invocation")
    parser.add_argument("--validate-controls", action="store_true", help="Tiny repeat/zero-recurrence controls; automatic for smoke tests")
    parser.add_argument("--ff-reference", default="", help="Existing complete baseline FF events.npz, recorded and verified")
    parser.add_argument("--compare-runs", nargs=2, metavar=("ADEX_DIR", "HH_DIR"), help="Validate matching baseline runs without simulation")
    return parser


def select_full(p, c, smoke=False, trial_ids=""):
    with p.tables.open_file(p.locate_data(c), mode="r") as f:
        labels = np.asarray(f.root.labels[:], dtype=np.int64)
        eligible = np.flatnonzero((labels >= 0) & (labels <= 9)).astype(np.int64)
        if smoke:
            if trial_ids:
                ids = np.asarray([int(x) for x in trial_ids.split(",")], dtype=np.int64)
                if not 1 <= len(ids) <= 2 or len(set(ids)) != len(ids) or not np.isin(ids, eligible).all():
                    raise ValueError("Smoke IDs must be 1–2 distinct English source trials in this split")
            else:
                small = replace(c, all_trials=False, trials_per_digit=1, digits="0,1")
                ids, _, _ = p.select_trials(f, small, p.seeds_for(c)["selection"])
        else:
            if trial_ids:
                raise ValueError("--trial-ids requires --smoke-test; full mode never subsamples")
            ids = eligible
        if not len(ids):
            raise ValueError("No English trials in this split")
        try:
            speakers = np.asarray(f.root.extra.speaker[:], dtype=np.int64)[ids]
        except p.tables.NoSuchNodeError:
            speakers = np.full(len(ids), -1, dtype=np.int64)
        return ids, labels[ids], speakers, len(eligible), len(labels)


def source_info(p, entry):
    files = [Path(entry).resolve(), Path(__file__).resolve(), Path(p.__file__).resolve()]
    course = p.COURSE / "src/compbio2026/spiking.py"
    if course.exists():
        files.append(course)
    return {f.name: digest_file(f) for f in files}


def model_setup(p, c, inputs, rec, initial, conditions):
    if p.MODEL_NAME == "AdEx":
        conversion, compatibility = p.adex_compatibility()
        build = lambda name: p.build_network(c, inputs, rec, initial, name, conversion)
        checks = p.validate_primitives(c, conversion)
    else:
        compatibility = p.hh_implementation_info()
        build = lambda name: p.build_network(c, inputs, rec, initial, name)
        checks = p.validate_primitives(c)
    models = {name: build(name) for name in conditions}
    checks.update(p.validate_structure(models, inputs, rec, initial, c))
    return models, build, compatibility, checks


def pack_chunk(p, block, conditions):
    out = {"trial_ids": np.asarray([x["id"] for x in block], dtype=np.int64),
           "stimulus_sha256": np.asarray([x["hash"] for x in block]),
           "metrics_json": np.asarray([json.dumps(x["rows"], allow_nan=False) for x in block])}
    for name in conditions:
        out[name + "_valid"] = np.asarray([x["valid"][name] for x in block], dtype=bool)
        for suffix in ("", "_pre"):
            events = [x["events"][name + suffix] for x in block]
            t, u, offsets = p.pack_events(events)
            out.update({name + suffix + "_times_ms": t, name + suffix + "_neuron_ids": u,
                        name + suffix + "_offsets": offsets})
    return out


def commit_block(p, outdir, meta, block):
    if not block:
        return
    arrays = pack_chunk(p, block, meta["conditions"])
    start = meta["committed_trials"]
    np.testing.assert_array_equal(arrays["trial_ids"], meta["trial_ids"][start:start + len(block)])
    filename = f"chunk_{start:06d}_{uuid.uuid4().hex[:12]}.npz"
    path = outdir / "checkpoints" / filename
    atomic_npz(path, arrays)
    updated = {**meta, "chunks": meta["chunks"] + [{"file": "checkpoints/" + filename,
                "start": start, "count": len(block), "sha256": digest_file(path)}],
               "committed_trials": start + len(block)}
    atomic_json(outdir / "metadata.json", updated)
    meta.update(updated)
    block.clear()


def checked_chunks(outdir, meta):
    cursor = 0
    for chunk in meta["chunks"]:
        path = (outdir / chunk["file"]).resolve()
        if path.parent != (outdir / "checkpoints").resolve():
            raise ValueError("Invalid checkpoint path")
        assert cursor == chunk["start"] and digest_file(path) == chunk["sha256"], "Checkpoint integrity failure"
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z["trial_ids"], meta["trial_ids"][cursor:cursor + chunk["count"]])
            for name in meta["conditions"]:
                assert z[name + "_valid"].shape == (chunk["count"],)
                for suffix in ("", "_pre"):
                    key = name + suffix
                    offsets = z[key + "_offsets"]
                    assert len(offsets) == chunk["count"] + 1 and offsets[0] == 0
                    assert np.all(np.diff(offsets) >= 0)
                    assert offsets[-1] == len(z[key + "_times_ms"]) == len(z[key + "_neuron_ids"])
            yield z
        cursor += chunk["count"]
    assert cursor == meta["committed_trials"]


def finalise(outdir, meta, p):
    """Stream event arrays via disk-backed .npy files; never collect a full event set in RAM."""
    count = meta["committed_trials"]
    with np.load(outdir / "shared.npz", allow_pickle=False) as shared:
        arrays = {k: shared[k] for k in shared.files}
    arrays["planned_trial_ids"] = arrays["trial_ids"].copy()
    for key in ("trial_ids", "labels", "speaker_ids", "trial_order", "source_split"):
        arrays[key] = arrays[key][:count]
    arrays.update(model=np.asarray(meta["model"]), condition=np.asarray(meta["conditions"]))
    names = [name + suffix for name in meta["conditions"] for suffix in ("", "_pre")]
    offsets = {name: [0] for name in names}
    valids = {name: [] for name in meta["conditions"]}
    hashes, rows = [], []
    for z in checked_chunks(outdir, meta):
        hashes.extend(z["stimulus_sha256"].tolist())
        for value in z["metrics_json"]:
            rows.extend(json.loads(str(value)))
        for name in names:
            base = offsets[name][-1]
            offsets[name].extend((z[name + "_offsets"][1:] + base).tolist())
        for name in valids:
            valids[name].extend(z[name + "_valid"].tolist())
    failures = [r for r in rows if not r["valid"]]
    meta["failures"] = failures
    meta["status"] = "complete" if count == len(meta["trial_ids"]) and not failures else "partially_valid"
    meta["validation"]["checkpoint_integrity"] = True
    meta["validation"]["offsets_validity_and_roster"] = True
    meta["archive_trial_count"] = count
    arrays["stimulus_sha256"] = np.asarray(hashes, dtype="U64")
    for name in valids:
        arrays[name + "_valid"] = np.asarray(valids[name], dtype=bool)
    for name in names:
        arrays[name + "_offsets"] = np.asarray(offsets[name], dtype=np.int64)
    arrays["metadata_json"] = np.asarray(json.dumps(meta, allow_nan=False))
    temporary = outdir / ("events." + uuid.uuid4().hex + ".tmp")
    with tempfile.TemporaryDirectory(prefix="export_", dir=outdir) as scratch:
        maps = {}
        try:
            for name in names:
                for suffix, dtype in (("_times_ms", np.float64), ("_neuron_ids", np.int32)):
                    key = name + suffix
                    path = Path(scratch) / (key + ".npy")
                    if offsets[name][-1]:
                        maps[key] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(offsets[name][-1],))
                    else:
                        np.save(path, np.empty(0, dtype=dtype), allow_pickle=False)
            positions = {name: 0 for name in names}
            for z in checked_chunks(outdir, meta):
                for name in names:
                    n = len(z[name + "_times_ms"])
                    for suffix in ("_times_ms", "_neuron_ids"):
                        if n:
                            maps[name + suffix][positions[name]:positions[name] + n] = z[name + suffix]
                    positions[name] += n
            for a in maps.values():
                a.flush()
            with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for key, value in arrays.items():
                    assert not value.dtype.hasobject
                    with archive.open(key + ".npy", "w", force_zip64=True) as member:
                        np.lib.format.write_array(member, value, allow_pickle=False)
                for path in sorted(Path(scratch).glob("*.npy")):
                    archive.write(path, path.name)
        finally:
            for a in maps.values():
                a._mmap.close()
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None, "Archive CRC check failed"
    with np.load(temporary, allow_pickle=False) as z:
        for key, value in arrays.items():
            np.testing.assert_array_equal(z[key], value, err_msg=key)
    os.replace(temporary, outdir / "events.npz")
    p.write_csv_checked(outdir / "trials.csv", rows)
    meta["validation"]["archive_crc_and_small_array_reload"] = True
    meta["archive_sha256"] = digest_file(outdir / "events.npz")
    atomic_json(outdir / "metadata.json", meta)
    marker = "COMPLETE.txt" if meta["status"] == "complete" else "PARTIAL.txt"
    (outdir / marker).write_text(f"Full simulation: {meta['status']}; {count}/{len(meta['trial_ids'])} attempted trials. Check condition_valid masks.\n")
    if marker == "COMPLETE.txt":
        (outdir / "PARTIAL.txt").unlink(missing_ok=True)


def validate_match(left_dir, right_dir, cross_model=True):
    left_dir, right_dir = Path(left_dir), Path(right_dir)
    metas = [json.loads((d / "metadata.json").read_text()) for d in (left_dir, right_dir)]
    if any(m["status"] != "complete" for m in metas):
        raise ValueError("Comparison requires two complete, numerically valid runs")
    if cross_model:
        assert [m["model"] for m in metas] == ["AdEx", "HH"]
        assert metas[0]["config"]["input_g"] == 3e-5 and metas[1]["config"]["input_g"] == 3e-4
    else:
        assert metas[0]["model"] == metas[1]["model"] and "FF" in metas[0]["conditions"]
    keys = ["trial_ids", "labels", "speaker_ids", "trial_order", "source_split", "input_connectivity",
            "initial_voltage_mv", "receptive_field_centres", "stimulus_sha256", "neuron_ids", "population"]
    if cross_model:
        keys.append("recurrent_connectivity")
    with np.load(left_dir / "events.npz", allow_pickle=False) as a, np.load(right_dir / "events.npz", allow_pickle=False) as b:
        for key in keys:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    exempt = SKIP_CONFIG | {"conditions", "output_dir", "data_path", "diagnostic_trials"}
    exempt |= {"input_g", "threshold_mv"} if cross_model else REC_ONLY
    common = metas[0]["config"].keys() & metas[1]["config"].keys()
    for key in common - exempt:
        assert metas[0]["config"][key] == metas[1]["config"][key], key
    assert metas[0]["geometry"] == metas[1]["geometry"]
    assert metas[0]["source_hdf5"]["sha256"] == metas[1]["source_hdf5"]["sha256"]
    assert metas[0]["seeds"] == metas[1]["seeds"]
    return {"passed": True, "shared_arrays": keys, "matched_config_fields": sorted(common - exempt),
            "intentional_differences": "intrinsic models and input_g; output spikes not compared" if cross_model else "recurrent parameters", "directories": [str(left_dir), str(right_dir)]}


def compare_zero_recurrence(p, left, right):
    if p.MODEL_NAME != "HH":
        return p.compare_runs(left, right, "full zero recurrence", atol=1e-8)
    # Distinct XLA graph layouts amplify roundoff around HH upstrokes. Bound
    # voltage error to 0.1 uV and gate error to 1 ppm; spike times remain exact.
    np.testing.assert_allclose(left[0], right[0], rtol=0, atol=1e-4)
    np.testing.assert_array_equal(left[1], right[1])
    np.testing.assert_allclose(left[2], right[2], rtol=0, atol=1e-6)
    return {"passed": True, "voltage_atol_mv": 1e-4, "gate_atol": 1e-6,
            "max_voltage_difference_mv": float(np.max(np.abs(left[0]-right[0]))),
            "max_gate_difference": float(np.max(np.abs(left[2]-right[2]))),
            "spike_crossings_identical": True}


def compare_repeat(p, left, right, model, backend):
    """Check a repeated compiled trial, allowing only backend round-off.

    CPU repeats retain the prototype's bitwise control. GPU kernels can vary in
    the last floating-point bits between launches; voltage/state tolerances are
    explicit while event/crossing masks remain exact.
    """
    if backend == "cpu":
        return p.compare_runs(left, right, "full repeat", atol=0.)
    voltage_atol = 1e-6 if model == "AdEx" else 1e-4
    state_atol = 1e-8 if model == "AdEx" else 1e-6
    np.testing.assert_allclose(left[0], right[0], rtol=0., atol=voltage_atol, err_msg="full repeat")
    np.testing.assert_array_equal(left[1], right[1], err_msg="full repeat spike/crossing mask")
    np.testing.assert_allclose(left[2], right[2], rtol=0., atol=state_atol, err_msg="full repeat state")
    return {"passed": True, "max_voltage_difference_mv": float(np.max(np.abs(left[0] - right[0]))),
            "voltage_atol_mv": voltage_atol, "state_atol": state_atol,
            "spike_crossings_identical": True, "backend_roundoff_allowed": True}


def full_main(model, entry):
    p = prototype(model)
    args = full_parser(p, model).parse_args()
    if args.compare_runs:
        report = validate_match(*args.compare_runs)
        print(json.dumps(report, indent=2))
        return
    if args.resume and args.run_dir:
        raise ValueError("Use --resume or --run-dir, not both")
    if args.checkpoint_trials < 1 or (args.stop_after is not None and args.stop_after < 1):
        raise ValueError("Checkpoint/stop trial counts must be positive")
    config_names = {f.name for f in fields(p.Config)} - SKIP_CONFIG
    c = full_config(p, model)
    for key in config_names:
        setattr(c, key, getattr(args, key))
    saved = None
    if args.resume:
        outdir = Path(args.resume).expanduser().resolve()
        saved = json.loads((outdir / "metadata.json").read_text())
        assert saved["model"] == model and saved["run_type"] == "full"
        explicit = {x.split("=", 1)[0][2:].replace("-", "_") for x in sys.argv[1:] if x.startswith("--")}
        stored = p.Config(**saved["config"])
        for key in config_names & explicit:
            if key not in {"data_path", "output_dir", "diagnostic_trials"} and getattr(c, key) != getattr(stored, key):
                raise ValueError(f"Resume cannot change scientific parameter {key}")
        for key in {"data_path", "output_dir", "diagnostic_trials"} & explicit:
            setattr(stored, key, getattr(c, key))
        if "trial_ids" in explicit and args.trial_ids != saved["selection"]["explicit_trial_ids"]:
            raise ValueError("Resume cannot change trial selection")
        if "smoke_test" in explicit and args.smoke_test != saved["selection"]["smoke_test"]:
            raise ValueError("Resume cannot change full/smoke selection mode")
        c = stored
        args.smoke_test = saved["selection"]["smoke_test"]
        args.trial_ids = saved["selection"]["explicit_trial_ids"]
        args.validate_controls |= saved["validate_controls"]
        if not args.ff_reference:
            args.ff_reference = saved.get("ff_reference", "")
    p.check_config(c)
    args.validate_controls |= args.smoke_test
    ids, labels, speakers, eligible_count, total_count = select_full(p, c, args.smoke_test, args.trial_ids)
    selection = {"smoke_test": args.smoke_test, "explicit_trial_ids": args.trial_ids,
                 "policy": "explicit/deterministic two-trial smoke" if args.smoke_test else "All original labels 0–9, ascending source ID; no subsampling",
                 "english_trials_available": eligible_count, "total_source_trials": total_count}
    if args.plan_only:
        print(json.dumps({"model": model, "split": c.split, "selected_count": len(ids),
                          "digit_counts": {str(i): int((labels == i).sum()) for i in range(10)},
                          "selection": selection, "config": asdict(c)}, indent=2))
        return
    if not saved:
        outdir = Path(args.run_dir).expanduser().resolve() if args.run_dir else Path(c.output_dir).expanduser().resolve() / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{model}_full_" + uuid.uuid4().hex[:8])
        outdir.mkdir(parents=True, exist_ok=False)
        (outdir / "checkpoints").mkdir()
    with run_lock(outdir):
        # Re-read under the lock; another invocation may have completed since parsing.
        if saved:
            saved = json.loads((outdir / "metadata.json").read_text())
        start_time = time.perf_counter()
        data = p.locate_data(c)
        source = {"path": str(data), "size_bytes": data.stat().st_size,
                  "mtime_ns": data.stat().st_mtime_ns, "sha256": digest_file(data)}
        versions = {x: importlib.metadata.version(x) for x in ("jax", "jaxlib", "jaxley", "numpy", "tables", "matplotlib", "pandas")}
        hashes = source_info(p, entry)
        scientific = {k: v for k, v in asdict(c).items() if k not in SKIP_CONFIG | {"data_path", "output_dir", "diagnostic_trials"}}
        identity = {"model": model, "scientific_config": scientific, "source_sha256": source["sha256"],
                    "trial_ids": ids.tolist(), "code": hashes, "packages": versions,
                    "backend": p.jax.default_backend()}
        fingerprint = digest_json(identity)
        if saved:
            if saved["fingerprint"] != fingerprint:
                raise ValueError("Resume identity changed: data, trials, parameters, source code, packages or backend")
            if saved["status"] == "complete":
                assert digest_file(outdir / "events.npz") == saved["archive_sha256"]
                print(f"Already complete; no simulation repeated: {outdir}")
                return
        seeds = p.seeds_for(c)
        inputs, rec, provenance = p.connectivity(c, seeds)
        initial = np.random.default_rng(seeds["initial"]).normal(c.initial_mv, c.initial_sd_mv, c.n_out)
        if np.any(initial >= c.threshold_mv):
            raise ValueError("Initial voltages must be below the spike/crossing threshold")
        conditions = ["FF", "REC"] if c.conditions == "both" else [c.conditions]
        if args.ff_reference:
            ref = Path(args.ff_reference).expanduser().resolve()
            refmeta = json.loads((ref.parent / "metadata.json").read_text())
            assert ref.name == "events.npz" and refmeta["model"] == model and refmeta["status"] == "complete"
            assert "FF" in refmeta["conditions"] and refmeta["source_hdf5"]["sha256"] == source["sha256"]
            with np.load(ref, allow_pickle=False) as z:
                for key, expected in (("trial_ids", ids), ("labels", labels), ("speaker_ids", speakers),
                                      ("input_connectivity", inputs), ("initial_voltage_mv", initial)):
                    np.testing.assert_array_equal(z[key], expected, err_msg="FF reference: " + key)
        models, build, compatibility, checks = model_setup(p, c, inputs, rec, initial, conditions)
        shared = p.sweep_shared_arrays(c, ids, labels, speakers, inputs, rec, initial, models)
        if saved:
            meta = saved
            with np.load(outdir / "shared.npz", allow_pickle=False) as z:
                for key, value in shared.items():
                    np.testing.assert_array_equal(z[key], value, err_msg=key)
            for _ in checked_chunks(outdir, meta):
                pass
        else:
            details = p.make_sweep_metadata(c, models, seeds, data, shared, provenance, compatibility, checks)
            keep = ("neuron_parameters", "initial_neuronal_states", "geometry", "units", "initial_synapse_s",
                    "initial_input_leak_g", "reset_protocol", "replay_mapping", "replay_lag", "waveform_comparison_note")
            meta = {k: details[k] for k in keep}
            meta.update(schema_version=2, run_type="full", model=model, status="running", config=asdict(c),
                        selection=selection, trial_ids=ids.tolist(), labels=labels.tolist(), speaker_ids=speakers.tolist(),
                        conditions=conditions, seeds=seeds, source_hdf5=source, source_hashes=hashes,
                        script_sha256=hashes[Path(entry).name], package_versions=versions,
                        jax_devices=[str(d) for d in p.jax.devices()], backend=p.jax.default_backend(), jax_enable_x64=True,
                        fingerprint=fingerprint, compatibility=compatibility, input_wiring_implementation=provenance,
                        synapse_parameters={k: x["synapse_params"] for k, x in models.items()},
                        connectivity_hashes={k: p.array_hash(shared[k]) for k in ("input_connectivity", "recurrent_connectivity", "initial_voltage_mv")},
                        validate_controls=args.validate_controls, validation=checks, chunks=[], committed_trials=0,
                        failures=[], invocations=[], ff_reference=args.ff_reference,
                        spike_detection="AdEx_spikes at sample k" if model == "AdEx" else "Upward voltage crossings at configured threshold",
                        metric_definitions={"primary_window": "[0,input_ms)", "post_window": "[input_ms,input_ms+post_ms]",
                            "ISI": "Within-neuron, within-trial input-window intervals only; undefined if no intervals",
                            "failed_trial": "C_valid=False, empty event slice, null rates: missing data, never a silent observation",
                            "partial_archive": "Only committed prefix is exported; planned_trial_ids records all requested trials",
                            "stimulus_sha256": "SHA256 of contiguous float64 replay bytes, identical to prototypes"})
            atomic_npz(outdir / "shared.npz", shared)
            atomic_json(outdir / "metadata.json", meta)
        print(f"{model} {c.split}: {len(ids)} English trials; {meta['committed_trials']} committed; backend={meta['backend']}; {outdir}", flush=True)
        block = []
        first_hashes = meta["validation"].get("first_recording_hashes", {})
        first_wave = None
        try:
            with p.tables.open_file(data, mode="r") as f:
                begin = meta["committed_trials"]
                end = len(ids) if args.stop_after is None else min(len(ids), begin + args.stop_after)
                for order in range(begin, end):
                    trial = int(ids[order])
                    wave, replay_stats, raw, pulses = p.replay(f.root.spikes.times[trial], f.root.spikes.units[trial], c)
                    item = {"id": trial, "hash": hashlib.sha256(wave.tobytes()).hexdigest(), "events": {}, "valid": {}, "rows": []}
                    recordings = {}
                    for name in conditions:
                        row = {"trial_order": order, "trial_id": trial, "label": int(labels[order]), "speaker_id": int(speakers[order]),
                               "model": model, "condition": name, "source_split": c.split, "valid": False, "error": "",
                               "elapsed_seconds": 0., "spike_count": None, **dict.fromkeys(p.SWEEP_MEASUREMENTS),
                               **dict.fromkeys(p.POSTHOC_EXTRA), "replay_json": json.dumps(replay_stats)}
                        t0 = time.perf_counter()
                        try:
                            record = p.simulate(models[name], wave, c)
                        except FloatingPointError as error:
                            row["error"] = str(error)
                            output = pre = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int32))
                            print(f"NUMERICAL FAILURE {name} trial {trial}: {error}", flush=True)
                        else:
                            metrics, rates, voltage = p.sweep_measurement(record, c)
                            extra, _ = p.posthoc_measurement(record, rates, voltage, c)
                            row.update(metrics, **extra, valid=True, spike_count=int(record[1][:, c.steps(c.settle_ms):c.steps(c.settle_ms+c.input_ms)].sum()))
                            output, pre = p.extract_events(record[1], c)
                            recordings[name] = record
                            if order == 0 and args.validate_controls:
                                repeat = p.simulate(models[name], wave, c)
                                meta["validation"][name + "_repeat"] = compare_repeat(p, record, repeat, model, meta["backend"])
                                first_hashes[name] = p.recording_hash(record)
                        row["elapsed_seconds"] = time.perf_counter() - t0
                        item["events"][name] = output
                        item["events"][name + "_pre"] = pre
                        item["valid"][name] = row["valid"]
                        item["rows"].append(row)
                    if order == 0 and args.validate_controls:
                        try:
                            ff = recordings.get("FF")
                            if ff is None:
                                ff = p.simulate(models.get("FF") or build("FF"), wave, c)
                            zero = p.simulate(models.get("REC") or build("REC"), wave, c, recurrent_scale=0.)
                            meta["validation"]["zero_recurrence"] = compare_zero_recurrence(p, ff, zero)
                        except FloatingPointError as error:
                            meta["validation"]["zero_recurrence"] = {"passed": False, "numerical_error": str(error)}
                        meta["validation"]["first_recording_hashes"] = first_hashes
                    if order < c.diagnostic_trials and recordings:
                        p.plot_trial(outdir / f"diagnostic_{order:06d}_{trial}", raw, pulses, recordings, c, f"{model} {c.split} source trial {trial}")
                    block.append(item)
                    if len(block) >= args.checkpoint_trials:
                        commit_block(p, outdir, meta, block)
                    print(f"{order+1}/{len(ids)} source={trial}; " + "; ".join(f"{r['condition']} {r['mean_rate_hz']} Hz valid={r['valid']}" for r in item["rows"]), flush=True)
                commit_block(p, outdir, meta, block)
                if args.validate_controls and first_hashes and meta["committed_trials"] == len(ids):
                    first_wave, _, _, _ = p.replay(f.root.spikes.times[int(ids[0])], f.root.spikes.units[int(ids[0])], c)
                    for name, expected in first_hashes.items():
                        assert p.recording_hash(p.simulate(models[name], first_wave, c)) == expected, "Reset after intervening trials changed output"
                    meta["validation"]["reset_after_intervening_trials"] = {"passed": True, "intervening_trials": len(ids)-1}
            meta["invocations"].append({"new_trials": meta["committed_trials"] - begin,
                "elapsed_seconds": time.perf_counter()-start_time, "source_path": str(data),
                "checkpoint_trials": args.checkpoint_trials, "diagnostic_trials": c.diagnostic_trials,
                "stop_after": args.stop_after, "devices": [str(d) for d in p.jax.devices()]})
            finalise(outdir, meta, p)
            if args.ff_reference and meta["status"] == "complete":
                meta["validation"]["FF_reference"] = validate_match(Path(args.ff_reference).resolve().parent, outdir, cross_model=False)
                atomic_json(outdir / "metadata.json", meta)
            print(f"{meta['status']}: {outdir / 'events.npz'}", flush=True)
        except BaseException as error:
            # Only prior committed chunks are authoritative; discard no user data.
            atomic_json(outdir / ("ERROR_" + uuid.uuid4().hex[:8] + ".json"),
                        {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
                         "resume": "Only committed journal chunks will be reused; uncommitted chunks are ignored"})
            committed = json.loads((outdir / "metadata.json").read_text())
            committed["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "error"
            atomic_json(outdir / "metadata.json", committed)
            (outdir / "COMPLETE.txt").unlink(missing_ok=True)
            (outdir / "PARTIAL.txt").write_text("Interrupted/error. Resume uses committed metadata.json checkpoints.\n")
            raise
        if meta["failures"]:
            raise SystemExit(2)  # Allows scheduler afterok dependencies to reject invalid FF references.


def build_manifest(model, base, seeds, enable_fan_in=False, enable_width=False, rerun_ff=False):
    """Eleven active OFAT configurations, plus explicitly enabled input deviations."""
    active = ["rec_probability", "rec_gain", "EI_ratio"]
    if enable_fan_in:
        active.append("fan_in")
    if enable_width:
        active.append("width")
    entries = [("baseline", "baseline", None)]
    for parameter in active:
        for value in GRIDS[parameter]:
            if value != BASE_VALUES[parameter]:
                label = f"{value:.2f}" if parameter == "rec_probability" else f"{value:g}"
                entries.append((parameter + "_" + label.replace(".", "p"), parameter, value))
    rows = []
    for config_id, parameter, value in entries:
        config = dict(base)
        gain = 1.
        if parameter == "rec_gain":
            gain = value
            config["rec_exc_g"] = 3e-5 * value
            config["rec_inh_g"] = 1.2e-4 * value
        elif parameter == "EI_ratio":
            config["rec_inh_g"] = 3e-5 * value
        elif parameter != "baseline":
            config[parameter] = value
        changed = {k for k in base if config[k] != base[k]}
        permitted = {"rec_gain": {"rec_exc_g", "rec_inh_g"}, "EI_ratio": {"rec_inh_g"},
                     "baseline": set()}.get(parameter, {parameter})
        assert changed == permitted, (config_id, changed, permitted)
        both = rerun_ff or parameter in {"fan_in", "width"}
        config["conditions"] = "both" if both else "REC"
        output = config_id
        reference = output + "/events.npz" if both else "baseline_FF/events.npz"
        rows.append({"config_id": config_id, "model": model, "parameter_name": parameter,
                     "parameter_value": value, "rec_probability": config["rec_probability"], "rec_gain": gain,
                     "rec_exc_g": config["rec_exc_g"], "rec_inh_g": config["rec_inh_g"],
                     "EI_ratio": config["rec_inh_g"] / config["rec_exc_g"],
                     "fan_in": config["fan_in"], "width": config["width"], "input_g": config["input_g"],
                     "condition_to_run": config["conditions"], "is_baseline": parameter == "baseline",
                     "FF_reference": reference, "output_directory": output,
                     "seed": config["seed"], **{k+"_seed": v for k, v in seeds.items()},
                     "full_parameters": config})
    expected = 11 + 3 * int(enable_fan_in) + 3 * int(enable_width)
    assert len(rows) == expected
    assert len({digest_json(r["full_parameters"]) for r in rows}) == expected
    return {"schema_version": 1, "model": model, "method": "one factor at a time; no factorial combinations",
            "baseline_input_g": base["input_g"], "active_parameters": active,
            "disabled_parameters": {k: v for k, v in GRIDS.items() if k not in active},
            "unique_configurations": expected, "rerun_ff": rerun_ff, "configurations": rows,
            "seeds": seeds, "full_baseline": base,
            "FF_reference_job": None if rerun_ff else {"config_id": "baseline_FF", "output_directory": "baseline_FF", "condition": "FF"},
            "EI_ratio_note": "Excitation is fixed; changing inhibition changes both ratio and total recurrent conductance",
            "selection": "All English trials in requested split, ascending source IDs, no subsampling",
            "paths": "All output paths relative to the manifest directory; source entry points resolved alongside this driver"}


def sweep_command(manifest, row, directory, resume=False):
    parameters = row["full_parameters"]
    script = ROOT / ("simulate_ff_rec_adex_full.py" if manifest["model"] == "AdEx" else "simulate_ff_rec_hh_full.py")
    command = [sys.executable, str(script)]
    for key, value in parameters.items():
        if key in SKIP_CONFIG | {"output_dir"} or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    target = Path(directory).resolve() / row["output_directory"]
    command.extend(["--resume" if resume else "--run-dir", str(target)])
    if row["condition_to_run"] == "REC":
        command.extend(["--ff-reference", str(Path(directory).resolve() / row["FF_reference"])])
    return command


def sweep_main(model, entry):
    parser = argparse.ArgumentParser(description=f"{model} OFAT manifest/job-array driver. Generation/dry-run is the default; execution requires --execute and a selector.")
    parser.add_argument("--manifest-dir", help="Existing manifest directory for array tasks")
    parser.add_argument("--output-dir", default=str(ROOT / "results"), help="Root for a new unique manifest directory")
    parser.add_argument("--data-path", default=str(prototype(model).COURSE / "data"))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--seed", type=int, default=2026)
    for key in ("selection", "input", "recurrent", "initial"):
        parser.add_argument("--" + key + "-seed", type=int)
    parser.add_argument("--enable-fan-in", action="store_true", help="Enable three input fan-in deviations in a NEW plan")
    parser.add_argument("--enable-width", action="store_true", help="Enable three input-width deviations in a NEW plan")
    parser.add_argument("--rerun-ff", action="store_true", help="New plan: both conditions in every configuration")
    parser.add_argument("--list-configs", action="store_true")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--config-id")
    selector.add_argument("--task-index", type=int, help="Zero-based configuration index: default 0..10")
    selector.add_argument("--ff-only", action="store_true", help="One shared baseline FF job")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually launch the selected full-data job (HPC only)")
    parser.add_argument("--resume", action="store_true", help="Explicitly resume selected job")
    args = parser.parse_args()
    if args.manifest_dir:
        directory = Path(args.manifest_dir).expanduser().resolve()
        manifest = json.loads((directory / "sweep_manifest.json").read_text())
        if manifest["model"] != model:
            raise ValueError("Manifest model disagrees with driver")
        generation_flags = {"--enable-fan-in", "--enable-width", "--rerun-ff", "--split", "--seed", "--data-path",
                            "--selection-seed", "--input-seed", "--recurrent-seed", "--initial-seed"}
        if any(x.split("=", 1)[0] in generation_flags for x in sys.argv[1:]):
            raise ValueError("Existing manifests are immutable; omit generation parameters")
        stored_hash = manifest.pop("manifest_sha256")
        assert digest_json(manifest) == stored_hash, "Manifest modified"
        manifest["manifest_sha256"] = stored_hash
    else:
        p = prototype(model)
        c = full_config(p, model)
        c.data_path, c.split, c.seed = args.data_path, args.split, args.seed
        for key in ("selection", "input", "recurrent", "initial"):
            setattr(c, key + "_seed", getattr(args, key + "_seed"))
        manifest = build_manifest(model, asdict(c), p.seeds_for(c), args.enable_fan_in, args.enable_width, args.rerun_ff)
        manifest["source_hashes"] = source_info(p, entry)
        manifest["manifest_sha256"] = digest_json(manifest)
        directory = Path(args.output_dir).expanduser().resolve() / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{model}_parameter_sweep_" + uuid.uuid4().hex[:8])
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "job_status").mkdir()
        atomic_json(directory / "sweep_manifest.json", manifest)
        csv_rows = [{k: (json.dumps(v, sort_keys=True) if isinstance(v, dict) else v) for k, v in row.items()}
                    for row in manifest["configurations"]]
        p.write_csv_checked(directory / "sweep_manifest.csv", csv_rows)
        index = {r["config_id"]: {"archive": r["output_directory"] + "/events.npz", "FF_reference": r["FF_reference"],
                                  "status_file": "job_status/" + r["config_id"] + ".json"} for r in manifest["configurations"]}
        if manifest["FF_reference_job"]:
            index["baseline_FF"] = {"archive": "baseline_FF/events.npz", "status_file": "job_status/baseline_FF.json"}
        atomic_json(directory / "sweep_index.json", index)
    print(f"Manifest: {directory}; {manifest['unique_configurations']} unique configurations", flush=True)
    configs = manifest["configurations"]
    if args.list_configs or not (args.config_id is not None or args.task_index is not None or args.ff_only):
        for i, row in enumerate(configs):
            print(f"{i:2d} {row['config_id']:25s} {row['condition_to_run']:4s} FF={row['FF_reference']}")
    row = None
    if args.ff_only:
        if not manifest["FF_reference_job"]:
            raise ValueError("This --rerun-ff manifest has no standalone FF job")
        row = {"config_id": "baseline_FF", "output_directory": "baseline_FF", "condition_to_run": "FF",
               "full_parameters": {**manifest["full_baseline"], "conditions": "FF"}}
    elif args.task_index is not None:
        if not 0 <= args.task_index < len(configs):
            raise ValueError(f"task-index must be in 0..{len(configs)-1}")
        row = configs[args.task_index]
    elif args.config_id is not None:
        row = next((r for r in configs if r["config_id"] == args.config_id), None)
        if row is None:
            raise ValueError("Unknown configuration ID")
    if args.execute and row is None:
        raise ValueError("Execution requires --config-id, --task-index or --ff-only; no implicit full sweep")
    if row:
        command = sweep_command(manifest, row, directory, args.resume)
        print(json.dumps({"argv": command, "executed": bool(args.execute and not args.dry_run)}, indent=2))
        if args.execute and not args.dry_run:
            # Child full runner checks data/config/code on resume and owns its output lock.
            status_path = directory / "job_status" / (row["config_id"] + ".json")
            t0 = time.perf_counter()
            result = subprocess.run(command, check=False)
            archive_dir = directory / row["output_directory"]
            state = json.loads((archive_dir / "metadata.json").read_text())["status"] if (archive_dir / "metadata.json").exists() else "error"
            atomic_json(status_path, {"config_id": row["config_id"], "exit_code": result.returncode, "status": state,
                                      "elapsed_seconds": time.perf_counter()-t0, "argv": command,
                                      "archive": row["output_directory"] + "/events.npz"})
            if result.returncode:
                raise SystemExit(result.returncode)
