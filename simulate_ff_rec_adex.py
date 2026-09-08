r"""Replay raw SHD spikes through matched feedforward and recurrent AdEx networks.

Run with the existing environment: .venv\Scripts\python.exe simulate_ff_rec_adex.py
Use --help, --smoke-test (two training trials), or --validate (additional controls).
--sweep-input-g evaluates a fixed subset at seven input conductances; customise
with --input-g-values "3e-5,3e-4". The notebook's G_SYN is exactly input_g (uS
per input synapse), not an independent parameter. Recurrent strengths stay fixed.
Sweep CSVs use [0,input_ms), with separate pre/post statistics. The pooled median
voltage is exact, calculated with a disk-backed temporary array; full-dataset
traces are never collected in RAM. --save-sweep-events saves ordinary-schema
event archives per point. Failed numerical evaluations remain explicit missing
values, never zero-rate observations. No value is selected as a new default.
--posthoc-input-g evaluates model-specific candidates on exactly five TRAIN trials
per English digit (50 total), with validation and event archives enabled.
--compare-posthoc-dir points to the completed other-model post-hoc directory.
No downloads, learning, decoding, or course-repository modifications are performed.
FF: BC -> E and I, without downstream connections. REC adds E->E, E->I,
I->E and I->I. In FF, I denotes the matched population; it exerts no inhibition.

Storage: each execution creates results/<UTC timestamp>_<UUID>/events.npz.
Load with np.load(path, allow_pickle=False). For condition C (FF or REC), trial k
occupies slice(C_offsets[k], C_offsets[k+1]) of C_times_ms and C_neuron_ids.
The complete roster is neuron_ids/population; a silent neuron still has an ID.
Primary spikes have 0 <= time <= input_ms + post_ms; pre-stimulus spikes are in
C_pre_times_ms / C_pre_neuron_ids / C_pre_offsets and have negative times.
metadata_json is a Unicode scalar, also written as readable metadata.json.
Connection matrices use [presynaptic, postsynaptic] order. Input IDs are always
0..699; output IDs are 0..n_exc+n_inh-1 (E first). No pickle/object arrays.

To use the course analysis later, split event arrays using offsets, divide times
by 1000, and build compbio2026.data.SHD(times, units, labels, speaker). Supply
n_channels=len(neuron_ids) explicitly to build_design_matrix (default is 700).
The closed final simulation endpoint is saved; a half-open analysis window will
exclude events exactly at its endpoint. No filtering is performed in this file.

References inspected (local checkout lacks spiking.py and notebooks 08/09):
https://github.com/CNNC-Lab/compbio-2026/tree/c7d722328959b7ff2ed7abc37278b0fbb5236929
  src/compbio2026/{data,spiking}.py; notebooks/08_full_spiking_model.ipynb;
  notebooks/09_state_matrix.ipynb (AdEx input gS=3e-5 uS).
https://github.com/CNNC-Lab/computational-biology-2025/blob/main/tutorial_example_P1_2_NEST3.ipynb
  E/I architecture only: no NEST current weights or delays are transferred.

Compatibility: Jaxley 0.14.0 AdEx updates voltage directly, omitting the factor
1000 converting mA/cm2 to uA/cm2 against capacitance in uF/cm2. A numerical
probe selects a minimal unit wrapper below; physical ADEX_PARAMS stay intact.
It also resets immediately, unlike the cap described in the course notebook.
Consequently its AdEx_spikes flag is authoritative, and -20 mV crossings are
diagnostic only. Recurrence uses its actual evolving presynaptic voltage;
there is no imposed output AP waveform, conduction delay, or refractory period.
All numerical choices are provisional, not validated biological parameters.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
import csv
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import sys
import tempfile
import time
import uuid

import numpy as np
import tables
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxley as jx
from jaxley.channels import AdEx, Leak
from jaxley.connect import connectivity_matrix_connect
from jaxley.synapses import IonotropicSynapse

ROOT = Path(__file__).resolve().parent
COURSE = ROOT if (ROOT / "src/compbio2026/spiking.py").is_file() else ROOT / "compbio-2026"
N_INPUT = 700
MODEL_NAME = "AdEx"
ADEX_PARAMS = {"AdEx_g_L": 5e-5, "AdEx_a": 1e-5,
               "AdEx_b": 0.0, "AdEx_tau_w": 100.0}
REFERENCE_COMMIT = "c7d722328959b7ff2ed7abc37278b0fbb5236929"


@dataclass
class Config:
    conditions: str = "both"
    split: str = "train"
    data_path: str = str(COURSE / "data")
    classes: str = "english"
    trials_per_digit: int = 2
    all_trials: bool = False
    digits: str = ""
    n_exc: int = 80
    n_inh: int = 20
    seed: int = 2026
    selection_seed: int | None = None
    input_seed: int | None = None
    recurrent_seed: int | None = None
    initial_seed: int | None = None
    wiring: str = "tonotopic"
    fan_in: int = 60
    width: float = 35.0
    rec_probability: float = 0.1
    input_g: float = 3e-5
    rec_exc_g: float = 3e-5
    rec_inh_g: float = 1.2e-4
    input_tau_ms: float = 5.0
    rec_exc_tau_ms: float = 5.0
    rec_inh_tau_ms: float = 5.0
    exc_reversal_mv: float = 0.0
    inh_reversal_mv: float = -80.0
    input_vth_mv: float = -35.0
    rec_vth_mv: float = -35.0
    input_delta_mv: float = 2.0
    rec_delta_mv: float = 2.0
    dt_ms: float = 0.1
    input_ms: float = 800.0
    post_ms: float = 100.0
    settle_ms: float = 50.0
    pulse_ms: float = 1.0
    replay_rest_mv: float = -70.0
    replay_spike_mv: float = 20.0
    initial_mv: float = -70.0
    initial_sd_mv: float = 0.0
    initial_w: float = 0.0
    adex_g_l: float = 5e-5
    adex_a: float = 1e-5
    adex_b: float = 0.0
    adex_tau_w: float = 100.0
    threshold_mv: float = -20.0
    output_dir: str = str(ROOT / "results")
    diagnostic_trials: int = 1
    smoke_test: bool = False
    validate: bool = False
    no_jit: bool = False
    sweep_input_g: bool = False
    input_g_values: str = "1e-5,3e-5,1e-4,3e-4,1e-3,1e-2,3e-2"
    save_sweep_events: bool = False
    posthoc_input_g: bool = False
    compare_posthoc_dir: str = ""

    @property
    def n_out(self):
        return self.n_exc + self.n_inh

    def steps(self, duration):
        return int(round(duration / self.dt_ms))

    @property
    def n_steps(self):
        return sum(self.steps(x) for x in (self.settle_ms, self.input_ms, self.post_ms))


def parse_config():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = Config()
    choices = {"conditions": ["FF", "REC", "both"], "split": ["train", "test"],
               "classes": ["english", "all"], "wiring": ["tonotopic", "random"]}
    for field in fields(Config):
        value = getattr(defaults, field.name)
        opts = {"default": value}
        if isinstance(value, bool):
            opts["action"] = "store_true"
        else:
            opts["type"] = int if value is None else type(value)
            opts["help"] = f"default: {value}" if value is not None else "derived from --seed"
        if field.name in choices:
            opts["choices"] = choices[field.name]
        parser.add_argument("--" + field.name.replace("_", "-"), **opts)
    cfg = Config(**vars(parser.parse_args()))
    configure_posthoc(cfg)
    if cfg.smoke_test:
        cfg.validate = True
        cfg.all_trials = False
        cfg.trials_per_digit = 1
        if not cfg.digits:
            cfg.digits = "0,1"
    check_config(cfg)
    return cfg


def check_config(c):
    for key, value in asdict(c).items():
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if min(c.n_exc, c.n_inh, c.trials_per_digit) < 1:
        raise ValueError("Both populations and trials_per_digit must be positive")
    if c.dt_ms <= 0 or c.input_ms <= 0 or c.settle_ms < c.dt_ms or c.post_ms < 0:
        raise ValueError("Require dt,input>0, settle>=dt, post>=0")
    for key in ("input_ms", "post_ms", "settle_ms", "pulse_ms"):
        value = getattr(c, key)
        if not np.isclose(value / c.dt_ms, round(value / c.dt_ms), atol=1e-9, rtol=0):
            raise ValueError(f"{key} must be an integer multiple of dt_ms")
    if c.pulse_ms < c.dt_ms or not 0 <= c.rec_probability <= 1:
        raise ValueError("Require pulse>=dt and recurrent probability in [0,1]")
    if not 1 <= c.fan_in <= N_INPUT or c.width <= 0:
        raise ValueError("Require fan-in in 1..700 and width>0")
    for key in ("input_tau_ms", "rec_exc_tau_ms", "rec_inh_tau_ms",
                "input_delta_mv", "rec_delta_mv", "adex_tau_w", "adex_g_l"):
        if getattr(c, key) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("input_g", "rec_exc_g", "rec_inh_g", "adex_a", "adex_b",
                "initial_sd_mv", "diagnostic_trials"):
        if getattr(c, key) < 0:
            raise ValueError(f"{key} must be nonnegative")
    if not -50 < c.threshold_mv < 0:
        raise ValueError("Diagnostic crossing threshold must be above v_T=-50 and below 0 mV")
    if c.replay_spike_mv <= c.replay_rest_mv:
        raise ValueError("Replay spike voltage must exceed rest")
    if c.sweep_input_g or c.posthoc_input_g:
        sweep_values(c)
        if c.all_trials:
            raise ValueError("Diagnostic sweeps use a subset: omit --all-trials")
    elif c.save_sweep_events:
        raise ValueError("--save-sweep-events requires --sweep-input-g")


def seeds_for(c):
    children = np.random.SeedSequence(c.seed).spawn(4)
    result = {}
    for name, child in zip(("selection", "input", "recurrent", "initial"), children):
        override = getattr(c, name + "_seed")
        result[name] = int(child.generate_state(1)[0]) if override is None else override
    return result


def locate_data(c):
    path = Path(c.data_path).expanduser().resolve()
    if path.is_file():
        expected = f"shd_{c.split}"
        if path.stem.startswith("shd_") and path.stem != expected:
            raise ValueError(f"File {path.name} disagrees with --split {c.split}")
        return path
    found = sorted(path.rglob(f"shd_{c.split}.h5"))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one shd_{c.split}.h5 under {path}; found {found}. No automatic download.")
    return found[0]


def select_trials(fh, c, seed):
    labels = np.asarray(fh.root.labels[:], dtype=np.int64)
    allowed = np.arange(10 if c.classes == "english" else 20)
    if c.digits:
        allowed = np.array(sorted(set(int(x) for x in c.digits.split(","))))
        if not np.all(np.isin(allowed, np.arange(10 if c.classes == "english" else 20))):
            raise ValueError("--digits must be original labels permitted by --classes")
    rng = np.random.default_rng(seed)
    groups = []
    for label in allowed:
        candidates = np.flatnonzero(labels == label)
        if len(candidates) == 0 or (not c.all_trials and len(candidates) < c.trials_per_digit):
            raise ValueError(f"Insufficient trials for original label {label}")
        groups.append(candidates if c.all_trials else rng.choice(candidates, c.trials_per_digit, replace=False))
    ids = rng.permutation(np.concatenate(groups)).astype(np.int64)
    try:
        speakers = np.asarray(fh.root.extra.speaker[:], dtype=np.int64)[ids]
    except tables.NoSuchNodeError:
        speakers = np.full(len(ids), -1, dtype=np.int64)
    return ids, labels[ids], speakers


def input_connectivity(c, seed):
    """Course utility, separately across the full axis for E and I.

    The self-contained fallback follows upstream spiking.py at REFERENCE_COMMIT;
    no runtime fetch/exec. Use the local course utility when it is available.
    """
    rng = np.random.default_rng(seed)
    utility = None
    if (COURSE / "src/compbio2026/spiking.py").exists():
        sys.path.insert(0, str(COURSE / "src"))
        from compbio2026 import spiking
        utility = getattr(spiking, c.wiring + "_connectivity")
    matrices = []
    for n in (c.n_exc, c.n_inh):
        if utility is not None:
            kw = {"fan_in": c.fan_in, "rng": rng}
            if c.wiring == "tonotopic":
                kw["width"] = c.width
            matrix = utility(N_INPUT, n, **kw)
        else:
            matrix = np.zeros((N_INPUT, n), dtype=bool)
            for col, centre in enumerate(np.linspace(0, N_INPUT - 1, n)):
                probabilities = None
                if c.wiring == "tonotopic":
                    probabilities = np.exp(-0.5 * ((np.arange(N_INPUT) - centre) / c.width) ** 2)
                    probabilities /= probabilities.sum()
                matrix[rng.choice(N_INPUT, c.fan_in, replace=False, p=probabilities), col] = True
        matrices.append(matrix)
    return np.concatenate(matrices, axis=1), "local course utility" if utility else "embedded course-equivalent fallback"


def connectivity(c, seeds):
    inputs, provenance = input_connectivity(c, seeds["input"])
    recurrent = np.random.default_rng(seeds["recurrent"]).random((c.n_out, c.n_out)) < c.rec_probability
    np.fill_diagonal(recurrent, False)
    assert np.all(inputs.sum(axis=0) == c.fan_in)
    return inputs, recurrent, provenance


def replay(times_seconds, units, c):
    """Ceil event times to the next grid point (never advance an input event).

    Accepted source window is [0,input_ms). Events whose ceil is input_ms are
    excluded explicitly. High pulses are half-open [start,start+pulse_ms),
    clipped at input offset. Same-channel overlapping OR adjacent high intervals
    merge in the voltage interface; report these separately from grid collisions.
    Clamp[k] appears in voltage sample k+1, so write at absolute_grid_index-1.
    Jaxley's synapse observes this clamp on the following integration step.
    """
    times = np.asarray(times_seconds, dtype=np.float64) * 1000.0
    raw_units = np.asarray(units)
    if times.shape != raw_units.shape or not np.isfinite(times).all():
        raise ValueError("Malformed input event arrays")
    if not np.all((raw_units >= 0) & (raw_units < N_INPUT) & (raw_units == np.floor(raw_units))):
        raise ValueError("SHD input channels must be integer IDs 0..699")
    units = raw_units.astype(np.int32)
    in_window = (times >= 0) & (times < c.input_ms)
    selected_t, selected_u = times[in_window], units[in_window]
    bins = np.ceil(selected_t / c.dt_ms).astype(np.int64)
    valid = bins < c.steps(c.input_ms)
    selected_t, selected_u, bins = selected_t[valid], selected_u[valid], bins[valid]
    wave = np.full((N_INPUT, c.n_steps), c.replay_rest_mv, dtype=np.float64)
    onset = c.steps(c.settle_ms)
    end = onset + c.steps(c.input_ms)
    width = c.steps(c.pulse_ms)
    duplicates = merged = truncated = 0
    pulse_channels, pulse_bins = [], []
    for unit in np.unique(selected_u):
        channel_bins = np.sort(bins[selected_u == unit])
        unique = np.unique(channel_bins)
        duplicates += len(channel_bins) - len(unique)
        previous_end = -1
        for start in unique:
            stop = min(int(start) + width, c.steps(c.input_ms))
            truncated += int(stop - start < width)
            if start <= previous_end:
                merged += 1
            else:
                pulse_channels.append(unit)
                pulse_bins.append(start)
            previous_end = max(previous_end, stop)
            wave[unit, onset + start - 1:onset + stop - 1] = c.replay_spike_mv
    assert np.all(wave[:, :onset - 1] == c.replay_rest_mv)
    assert np.all(wave[:, end - 1:] == c.replay_rest_mv)
    report = {
        "source_events": len(times), "excluded_input_window": int((~in_window).sum()),
        "excluded_ceil_at_offset": int((~valid).sum()), "accepted_events": len(bins),
        "same_channel_grid_collisions": int(duplicates),
        "additional_overlapping_or_adjacent_pulses": int(merged),
        "distinct_high_intervals": len(pulse_bins), "clipped_pulse_tails": int(truncated),
        "max_quantisation_delay_ms": float(np.max(bins * c.dt_ms - selected_t, initial=0)),
    }
    assert report["accepted_events"] == report["distinct_high_intervals"] + duplicates + merged
    return wave, report, (times, units), (np.asarray(pulse_bins) * c.dt_ms, np.asarray(pulse_channels, dtype=np.int32))


def adex_compatibility():
    """Fail closed on an unrecognised implementation; numerically verify units."""
    channel = AdEx(surrogate_warning=False)
    if "AdEx_spikes" not in channel.channel_states:
        raise RuntimeError("This script requires the inspected AdEx spike-state API")
    physical = {**channel.channel_params, **ADEX_PARAMS, "capacitance": 1.0}
    params = {k: jnp.array([float(v)]) for k, v in physical.items()}
    states = {k: jnp.array([float(v)]) for k, v in channel.channel_states.items()}
    voltage = jnp.array([-60.0])
    out = channel.update_states(states, 0.1, voltage, params)
    expected = .1 * 1000 * (-5e-5 * 10 + 5e-5 * 2 * np.exp(-5) - float(out["AdEx_w"][0]))
    observed = float(out["v"][0] + 60)
    if np.isclose(observed, expected, rtol=1e-5, atol=1e-10):
        factor = 1.0
    elif np.isclose(observed * 1000, expected, rtol=1e-5, atol=1e-10):
        factor = 1000.0
    else:
        raise RuntimeError(f"Unrecognised AdEx units: dv={observed}, expected={expected}")
    return factor, {"physical_probe_dv_mv": expected, "installed_probe_dv_mv": observed,
                    "internal_current_conversion_factor": factor,
                    "installed_adex_source_sha256": hashlib.sha256(inspect.getsource(AdEx).encode()).hexdigest()}


class CourseAdEx(AdEx):
    """Use installed AdEx dynamics with an explicitly probed current-unit adapter.

    Stored g_L/a are S/cm2, b/w are mA/cm2. Rescale temporary parameters and w
    for the direct update only, then restore w units. Reset and adaptation
    equations are the installed implementation, with identical E/I settings.
    """
    def __init__(self, conversion):
        super().__init__(name="AdEx", surrogate_warning=False)
        self.conversion = conversion

    def update_states(self, states, dt, v, params):
        p, s = dict(params), dict(states)
        for key in ("AdEx_g_L", "AdEx_a", "AdEx_b"):
            p[key] = p[key] * self.conversion
        s["AdEx_w"] = s["AdEx_w"] * self.conversion
        out = super().update_states(s, dt, v, p)
        out["AdEx_w"] = out["AdEx_w"] / self.conversion
        return out


def build_network(c, inputs, recurrent, initial, condition, conversion):
    cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
    net = jx.Network([cell] * (N_INPUT + c.n_out))
    pre = net.cell(list(range(N_INPUT)))
    post = net.cell(list(range(N_INPUT, N_INPUT + c.n_out)))
    pre.insert(Leak())
    post.insert(CourseAdEx(conversion))
    for key, value in {"radius": 10., "length": 10., "capacitance": 1., "v": c.replay_rest_mv}.items():
        net.set(key, value)
    # Jaxley views snapshot their columns: refresh after channel insertion.
    pre = net.cell(list(range(N_INPUT)))
    post = net.cell(list(range(N_INPUT, N_INPUT + c.n_out)))
    # Zero input leak makes the previous clamped voltage exact during synapse update.
    pre.set("Leak_gLeak", 0.0)
    post.set("v", initial)
    neuron_params = {**AdEx(surrogate_warning=False).channel_params,
                     "AdEx_g_L": c.adex_g_l, "AdEx_a": c.adex_a,
                     "AdEx_b": c.adex_b, "AdEx_tau_w": c.adex_tau_w}
    for key, value in neuron_params.items():
        post.set(key, value)
    post.set("AdEx_w", c.initial_w)
    post.set("AdEx_spikes", 0.0)
    specs = {"InputExc": (c.input_g, c.exc_reversal_mv, c.input_tau_ms, c.input_vth_mv, c.input_delta_mv)}
    connectivity_matrix_connect(pre, post, IonotropicSynapse("InputExc"), inputs)
    if condition == "REC":
        for name, start, stop, g, reversal, tau in (
            ("RecExc", 0, c.n_exc, c.rec_exc_g, c.exc_reversal_mv, c.rec_exc_tau_ms),
            ("RecInh", c.n_exc, c.n_out, c.rec_inh_g, c.inh_reversal_mv, c.rec_inh_tau_ms)):
            matrix = recurrent[start:stop]
            if matrix.any():
                connectivity_matrix_connect(net.cell(list(range(N_INPUT + start, N_INPUT + stop))),
                                            post, IonotropicSynapse(name), matrix)
                specs[name] = (g, reversal, tau, c.rec_vth_mv, c.rec_delta_mv)
    synaptic = {}
    for name, (g, reversal, tau, vth, delta) in specs.items():
        params = {f"{name}_gS": g, f"{name}_e_syn": reversal,
                  f"{name}_k_minus": 1 / tau, f"{name}_v_th": vth, f"{name}_delta": delta}
        for key, value in params.items():
            net.set(key, value)
        net.set(f"{name}_s", 0.0)  # Installed default 0.2 would cause a false onset transient.
        synaptic[name] = {**params, f"{name}_s_initial": 0.0}
        mask = net.edges["type"] == name
        for key, value in params.items():
            assert np.all(net.edges.loc[mask, key].to_numpy() == value)
            assert net.edges.loc[~mask, key].isna().all(), "Synapse names must isolate E/I parameters"
    post.record("v", verbose=False)
    post.record("AdEx_spikes", verbose=False)
    post.record("AdEx_w", verbose=False)
    # Bookkeeping remains concrete outside JIT; only waveform and recurrence scale vary.
    names, _, inds = pre.data_clamp("v", np.zeros((N_INPUT, 1)), verbose=False)
    templates = []
    for name in ("RecExc", "RecInh"):
        if name in specs:
            template = net.data_set(f"{name}_gS", specs[name][0], None)[0]
            templates.append(template)

    input_template = net.data_set("InputExc_gS", c.input_g, None)[0]

    def parameter_state(input_g=None, recurrent_scale=1.0):
        parameters = [{**t, "val": t["val"] * recurrent_scale} for t in templates]
        if input_g is not None:
            parameters.append({**input_template, "val": jnp.full_like(input_template["val"], input_g)})
        return parameters

    def run(waveform, recurrent_scale=1.0, input_g=None):
        parameters = parameter_state(input_g, recurrent_scale)
        # No t_max: the clamp defines an exact integer step count, avoiding float //.
        # No all_states: every call reconstructs initial voltage, w, flags and gates.
        return jx.integrate(net, data_clamps=(names, [waveform], inds),
                            param_state=parameters, delta_t=c.dt_ms)

    return {"net": net, "run": run if c.no_jit else jax.jit(run),
            "neuron_params": neuron_params, "synapse_params": synaptic,
            "parameter_state": parameter_state}


def simulate(model, waveform, c, recurrent_scale=1.0, input_g=None):
    if input_g is None:
        recordings = np.asarray(model["run"](jnp.asarray(waveform), recurrent_scale))
    else:
        recordings = np.asarray(model["run"](jnp.asarray(waveform), recurrent_scale,
                                              jnp.asarray(input_g, dtype=jnp.float64)))
    if recordings.shape != (3 * c.n_out, c.n_steps + 1):
        raise AssertionError(f"Unexpected recording shape {recordings.shape}")
    if not np.isfinite(recordings).all():
        raise FloatingPointError("Non-finite voltage, spike flag or adaptation state")
    v, flags, w = np.split(recordings, 3)
    if not np.isin(flags, [0, 1]).all() or flags[:, 0].any():
        raise AssertionError("Invalid/reset-carried spike flags")
    return v, flags.astype(bool), w


def extract_events(flags, c):
    # flags[:,0] is initial state; sample k is at absolute k*dt, not (k-1)*dt.
    neuron, sample = np.nonzero(flags)
    order = np.lexsort((neuron, sample))
    sample, neuron = sample[order], neuron[order].astype(np.int32)
    relative_steps = sample - c.steps(c.settle_ms)
    times = relative_steps.astype(np.float64) * c.dt_ms
    observation = relative_steps >= 0
    return (times[observation], neuron[observation]), (times[~observation], neuron[~observation])


def crossing_audit(v, flags, c):
    crossings = (v[:, 1:] > c.threshold_mv) & (v[:, :-1] <= c.threshold_mv)
    # Does each actual reset have exactly one crossing since its preceding reset?
    missing = multiple = trailing = 0
    latencies = []
    for neuron in range(c.n_out):
        cross_samples = np.flatnonzero(crossings[neuron]) + 1
        previous_reset = 0
        for reset in np.flatnonzero(flags[neuron]):
            between = cross_samples[(cross_samples > previous_reset) & (cross_samples <= reset)]
            missing += len(between) == 0
            multiple += len(between) > 1
            if len(between):
                latencies.append((reset - between[-1]) * c.dt_ms)
            previous_reset = reset
        trailing += int((cross_samples > previous_reset).sum())
    return {"threshold_crossings": int(crossings.sum()), "actual_spike_flags": int(flags.sum()),
            "resets_without_crossing": int(missing), "resets_with_multiple_crossings": int(multiple),
            "unfollowed_final_crossings": int(trailing),
            "max_crossing_to_reset_ms": float(max(latencies, default=0)),
            "authoritative_detector": "AdEx_spikes (immediate-reset implementation)"}


def diagnostics(v, flags, w, c):
    onset = c.steps(c.settle_ms)
    sample = np.arange(flags.shape[1])
    masks = {"pre": sample < onset,
             "input": (sample >= onset) & (sample < onset + c.steps(c.input_ms)),
             "post": sample >= onset + c.steps(c.input_ms),
             "observation": sample >= onset}
    durations = {"pre": c.settle_ms, "input": c.input_ms, "post": c.post_ms,
                 "observation": c.input_ms + c.post_ms}
    result = {"spike_detection": crossing_audit(v, flags, c)}
    for name, region in (("E", slice(0, c.n_exc)), ("I", slice(c.n_exc, c.n_out))):
        result[name] = {}
        for window, mask in masks.items():
            counts = flags[region][:, mask].sum(axis=1)
            duration = durations[window] / 1000
            result[name][window] = {
                "spikes": int(counts.sum()),
                "mean_rate_hz": float(counts.mean() / duration) if duration else None,
                "max_neuron_rate_hz": float(counts.max() / duration) if duration else None,
                "silent_fraction": float(np.mean(counts == 0))}
    tail_start = max(0, onset - c.steps(min(10., c.settle_ms)))
    result["settling"] = {
        "last_10ms_max_voltage_change_mv": float(np.max(np.abs(v[:, onset] - v[:, tail_start]))),
        "last_10ms_max_adaptation_change_ma_cm2": float(np.max(np.abs(w[:, onset] - w[:, tail_start]))),
        "voltage_at_onset_min_max_mv": [float(v[:, onset].min()), float(v[:, onset].max())],
        "pre_spikes": int(flags[:, :onset].sum())}
    result["voltage_min_max_mv"] = [float(v.min()), float(v.max())]
    intervals = [np.diff(np.flatnonzero(row)) * c.dt_ms for row in flags]
    intervals = np.concatenate([x for x in intervals if len(x)]) if any(len(x) for x in intervals) else np.array([])
    result["interspike_intervals"] = {
        "minimum_ms": float(intervals.min()) if len(intervals) else None,
        "median_ms": float(np.median(intervals)) if len(intervals) else None,
        "fraction_below_2ms": float(np.mean(intervals < 2.)) if len(intervals) else None,
        "note": "Installed AdEx has no explicit refractory period; assess fast firing before biological interpretation"}
    warnings = []
    for pop in ("E", "I"):
        stats = result[pop]["observation"]
        if stats["silent_fraction"] > .8:
            warnings.append(f"{pop}: >80% silent in this trial; inspect RF coverage and input strength")
        if stats["mean_rate_hz"] > 150 or stats["max_neuron_rate_hz"] > 300:
            warnings.append(f"{pop}: high firing rate; provisional conductances need inspection")
        post = result[pop]["post"]["mean_rate_hz"]
        if post is not None and post > 50:
            warnings.append(f"{pop}: post-input activity exceeds 50 Hz; inspect recurrence/tail")
    if result["settling"]["pre_spikes"] or result["settling"]["last_10ms_max_voltage_change_mv"] > .1:
        warnings.append("Pre-stimulus activity or voltage drift: settling may be inadequate")
    result["warnings"] = warnings
    return result


def edge_arrays(model):
    frame = model["net"].edges
    nodes = model["net"].nodes
    return {"pre_global_ids": nodes.loc[frame["pre_index"], "global_cell_index"].to_numpy(dtype=np.int32),
            "post_global_ids": nodes.loc[frame["post_index"], "global_cell_index"].to_numpy(dtype=np.int32),
            "synapse_types": frame["type"].to_numpy(dtype=str)}


def validate_structure(models, inputs, recurrent, initial, c):
    for name, model in models.items():
        arrays = edge_arrays(model)
        pre, post, types = (arrays[k] for k in ("pre_global_ids", "post_global_ids", "synapse_types"))
        pairs = np.column_stack((pre, post))
        assert len(np.unique(pairs, axis=0)) == len(pairs), "Duplicate connections"
        actual_input = np.zeros_like(inputs)
        mask = types == "InputExc"
        actual_input[pre[mask], post[mask] - N_INPUT] = True
        np.testing.assert_array_equal(actual_input, inputs)
        actual_recurrent = np.zeros_like(recurrent)
        mask = types != "InputExc"
        actual_recurrent[pre[mask] - N_INPUT, post[mask] - N_INPUT] = True
        np.testing.assert_array_equal(actual_recurrent, recurrent if name == "REC" else np.zeros_like(recurrent))
        assert not np.diag(actual_recurrent).any()
        nodes = model["net"].nodes.iloc[N_INPUT:]
        np.testing.assert_array_equal(nodes.v.to_numpy(), initial)
        assert np.all(nodes.AdEx_w == c.initial_w)
        assert np.all(nodes.AdEx_spikes == 0)
    if {"FF", "REC"}.issubset(models):
        assert models["FF"]["neuron_params"] == models["REC"]["neuron_params"]
        assert models["FF"]["synapse_params"]["InputExc"] == models["REC"]["synapse_params"]["InputExc"]
        # pandas equality handles absent-channel NaNs, including object/bool columns.
        assert models["FF"]["net"].nodes.equals(models["REC"]["net"].nodes)
    return {"matched_inputs_and_neurons": True, "recurrent_only_in_REC": True,
            "no_self_or_duplicate_connections": True,
            "recurrent_counts": {f"{a}_to_{b}": int(recurrent[s, t].sum())
                                 for a, s in (("E", slice(0, c.n_exc)), ("I", slice(c.n_exc, c.n_out)))
                                 for b, t in (("E", slice(0, c.n_exc)), ("I", slice(c.n_exc, c.n_out)))}}


def validate_primitives(c, conversion):
    tiny = Config(dt_ms=.1, input_ms=1., settle_ms=.2, post_ms=.2, pulse_ms=.2)
    # Same-bin collision, adjacent/overlapping pulses, both boundaries, tail clip.
    wave, report, _, _ = replay(np.array([-.0001, 0, .00011, .00012, .0002, .0009, .00099, .001]),
                                 np.zeros(8, dtype=int), tiny)
    assert report["excluded_input_window"] == 2 and report["excluded_ceil_at_offset"] == 1
    assert report["accepted_events"] == 5 and report["same_channel_grid_collisions"] == 2
    assert report["additional_overlapping_or_adjacent_pulses"] == 1
    assert report["distinct_high_intervals"] == 2 and report["clipped_pulse_tails"] == 1
    cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
    cell.insert(Leak()); cell.set("v", -70.); cell.record("v", verbose=False)
    clamp = cell.data_clamp("v", jnp.array([[-70., 20., -70.]]), verbose=False)
    result = np.asarray(jx.integrate(cell, data_clamps=clamp, delta_t=c.dt_ms))
    np.testing.assert_array_equal(result, [[-70., -70., 20., -70.]])
    flags = np.zeros((c.n_out, c.n_steps + 1), dtype=bool)
    onset = c.steps(c.settle_ms)
    flags[0, [onset - 1, onset, onset + 1, -1]] = True
    events, pre = extract_events(flags, c)
    np.testing.assert_allclose(events[0], [0., c.dt_ms, c.input_ms + c.post_ms], atol=1e-10)
    np.testing.assert_allclose(pre[0], [-c.dt_ms], atol=1e-10)
    # Verify corrected leak dynamics and immediate reset directly, including nonzero b/w.
    channel = CourseAdEx(conversion)
    p = {**channel.channel_params, **ADEX_PARAMS, "capacitance": 1.0, "AdEx_b": 2e-5}
    p = {k: jnp.array([float(v)]) for k, v in p.items()}
    states = {"AdEx_w": jnp.array([3e-5]), "AdEx_spikes": jnp.array([0.])}
    out = channel.update_states(states, .1, jnp.array([-60.]), p)
    expected_w = 3e-5 * np.exp(-.1 / 100) + 1e-4 * (1 - np.exp(-.1 / 100))
    expected_v = -60. + .1 * 1000 * (-5e-5 * 10 + 1e-4 * np.exp(-5) - expected_w)
    np.testing.assert_allclose(out["v"], expected_v, atol=1e-10)
    np.testing.assert_allclose(out["AdEx_w"], expected_w, atol=1e-14)
    reset = channel.update_states(states, .1, jnp.array([-10.]), p)
    assert float(reset["AdEx_spikes"][0]) == 1 and float(reset["v"][0]) == -58.
    reset_w = 3e-5 * np.exp(-.001) + 6e-4 * (1 - np.exp(-.001)) + 2e-5
    np.testing.assert_allclose(reset["AdEx_w"], reset_w, atol=1e-14)
    return {"replay_boundary_collision_tests": True, "recording_sample_clock": True,
            "spike_timestamp_and_endpoint_tests": True, "physical_AdEx_update_and_reset": True}


def compare_runs(left, right, label, atol=1e-8):
    np.testing.assert_allclose(left[0], right[0], rtol=0, atol=atol, err_msg=label)
    np.testing.assert_array_equal(left[1], right[1], err_msg=label)
    np.testing.assert_allclose(left[2], right[2], rtol=0, atol=1e-12, err_msg=label)
    return {"passed": True, "max_voltage_difference_mv": float(np.max(np.abs(left[0] - right[0]))),
            "voltage_atol_mv": atol, "spike_flags_identical": True}


def plot_trial(path, raw_input, pulse_events, recordings, c, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2 + len(recordings), 1, figsize=(12, 3 + 2.1 * len(recordings)), sharex=True)
    times, units = raw_input
    axes[0].scatter(times, units, s=.5, color="#333333", rasterized=True)
    axes[0].set(ylabel="SHD channel", title=title, ylim=(-5, 705))
    tt = (np.arange(c.n_steps + 1) - c.steps(c.settle_ms)) * c.dt_ms
    selected = np.unique(np.r_[np.linspace(0, c.n_exc - 1, 3, dtype=int),
                               np.linspace(c.n_exc, c.n_out - 1, 3, dtype=int)])
    for row, (name, (v, flags, w)) in enumerate(recordings.items(), start=1):
        output, pre = extract_events(flags, c)
        event_t = np.r_[pre[0], output[0]]
        event_u = np.r_[pre[1], output[1]]
        colours = np.where(event_u < c.n_exc, "#2878b5", "#dd6529")
        axes[row].scatter(event_t, event_u, s=5, marker="|", c=colours, linewidths=.6)
        axes[row].axhline(c.n_exc - .5, color="grey", lw=.5)
        axes[row].set(ylabel=f"{name} neuron", ylim=(-1, c.n_out))
        for unit in selected[:2]:
            axes[-1].plot(tt, v[unit], lw=.6, alpha=.7, label=f"{name} E{unit}")
        unit = selected[-1]
        axes[-1].plot(tt, v[unit], lw=.6, alpha=.7, label=f"{name} I{unit}")
    axes[-1].axhline(c.threshold_mv, color="grey", ls=":", lw=.7)
    axes[-1].set(ylabel="Voltage (mV)", xlabel="Time relative to input onset (ms)")
    axes[-1].legend(ncol=3, fontsize=7, loc="upper right")
    for ax in axes:
        ax.axvspan(-c.settle_ms, 0, color="#dddddd", alpha=.4)
        ax.axvline(0, color="grey", lw=.7)
        ax.axvline(c.input_ms, color="grey", ls="--", lw=.7)
        ax.set_xlim(-c.settle_ms, c.input_ms + c.post_ms)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=170)
    plt.close(fig)
    # A few complete trials only, never all-dataset voltage/clamp arrays.
    arrays = {"sample_times_ms": tt, "input_source_times_ms": times,
              "input_source_channel_ids": units, "replay_high_interval_start_ms": pulse_events[0],
              "replay_high_interval_channel_ids": pulse_events[1], "neuron_ids": np.arange(c.n_out)}
    for name, (v, flags, w) in recordings.items():
        arrays.update({f"{name}_voltage_mv": v, f"{name}_spike_flags": flags,
                       f"{name}_adaptation_ma_cm2": w})
    np.savez_compressed(path.with_suffix(".npz"), **arrays)


def pack_events(event_trials):
    offsets = np.r_[0, np.cumsum([len(x[0]) for x in event_trials])].astype(np.int64)
    return np.concatenate([x[0] for x in event_trials]), np.concatenate([x[1] for x in event_trials]), offsets


def save_results(outdir, arrays, metadata, c):
    """Verify every saved array and event roster without pickle, then mark complete."""
    metadata["validation"]["save_reload_all_arrays_and_silent_roster"] = True
    arrays["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=True, allow_nan=False))
    destination = outdir / "events.npz"
    # Output directory is exclusive. Exclusive file creation also prevents accidental overwrite.
    with destination.open("xb") as fh:
        np.savez_compressed(fh, **arrays)
    with np.load(destination, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        for key, expected in arrays.items():
            np.testing.assert_array_equal(loaded[key], expected, err_msg=key)
            assert loaded[key].dtype != object
        np.testing.assert_array_equal(loaded["neuron_ids"], np.arange(c.n_out))
        assert len(loaded["population"]) == c.n_out
        for name in metadata["conditions"]:
            for suffix in ("", "_pre"):
                offsets = loaded[f"{name}{suffix}_offsets"]
                times = loaded[f"{name}{suffix}_times_ms"]
                units = loaded[f"{name}{suffix}_neuron_ids"]
                assert len(offsets) == len(loaded["trial_ids"]) + 1
                assert offsets[0] == 0 and offsets[-1] == len(times) == len(units)
                assert np.all(np.diff(offsets) >= 0)
                assert np.isfinite(times).all() and np.all((units >= 0) & (units < c.n_out))
                if suffix:
                    assert np.all((times >= -c.settle_ms) & (times < 0))
                else:
                    assert np.all((times >= 0) & (times <= c.input_ms + c.post_ms + 1e-9))
                for start, stop in zip(offsets[:-1], offsets[1:]):
                    assert np.all(np.diff(times[start:stop]) >= 0)
        json.loads(str(loaded["metadata_json"]))
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    (outdir / "COMPLETE.txt").write_text("Simulation and archive reload validation completed.\n", encoding="utf-8")
    return destination



POSTHOC_EXTRA = ("max_neuron_rate_hz", "isi_count", "min_isi_ms", "median_isi_ms",
                 "isi_below_2ms_fraction")


def configure_posthoc(c):
    """A fixed balanced evaluation protocol; never changes ordinary defaults."""
    if not c.posthoc_input_g:
        if c.compare_posthoc_dir:
            raise ValueError("--compare-posthoc-dir requires --posthoc-input-g")
        return
    if c.all_trials or c.sweep_input_g or c.smoke_test:
        raise ValueError("--posthoc-input-g cannot be combined with --all-trials, --sweep-input-g or --smoke-test")
    c.split, c.classes, c.digits, c.trials_per_digit = "train", "english", "", 5
    c.input_g_values = "1e-5,3e-5,1e-4" if MODEL_NAME == "AdEx" else "1e-4,3e-4,1e-3"
    c.validate = True
    c.save_sweep_events = True


def validate_posthoc_selection(c, trial_ids, labels):
    assert c.split == "train" and c.classes == "english" and not c.all_trials
    assert len(trial_ids) == len(np.unique(trial_ids)) == 50
    counts = {int(d): int(np.count_nonzero(labels == d)) for d in range(10)}
    assert set(np.unique(labels)) == set(range(10)) and all(n == 5 for n in counts.values())
    return {"passed": True, "unique_trials": 50, "digit_counts": counts,
            "ordering": "Unchanged select_trials RNG permutation after balanced selection"}


def posthoc_measurement(record, rates, input_v, c):
    onset, offset = c.steps(c.settle_ms), c.steps(c.settle_ms + c.input_ms)
    intervals = [np.diff(np.flatnonzero(spikes)) * c.dt_ms
                 for spikes in record[1][:, onset:offset]]
    isi = np.concatenate(intervals)
    return {"max_neuron_rate_hz": float(rates.max()), "isi_count": int(isi.size),
            "min_isi_ms": float(isi.min()) if isi.size else None,
            "median_isi_ms": float(np.median(isi)) if isi.size else None,
            "isi_below_2ms_fraction": float(np.mean(isi < 2.)) if isi.size else None,
            "voltage_min_mv": float(input_v.min()), "voltage_max_mv": float(input_v.max())}, isi


def distribution(values, prefix):
    """Empirical trial distribution, sample SD; no independent-network inference."""
    a = np.asarray(values, dtype=float)
    names = ("mean", "median", "sd", "iqr", "p05", "p25", "p75", "p95", "min", "max")
    if not len(a):
        return {prefix + name: None for name in names}
    q05, q25, q75, q95 = np.percentile(a, [5, 25, 75, 95])
    vals = (a.mean(), np.median(a), a.std(ddof=1) if len(a) > 1 else None,
            q75-q25, q05, q25, q75, q95, a.min(), a.max())
    return {prefix + name: float(v) if v is not None else None for name, v in zip(names, vals)}


def posthoc_tables(rows, coarse, pooled_isi):
    summaries, digits = [], []
    for base in coarse:
        group = [r for r in rows if r["condition"] == base["condition"] and r["input_g_us"] == base["input_g_us"]]
        good = [r for r in group if r["valid"]]
        summary = dict(base)
        for metric, prefix in (("mean_rate_hz", "rate_hz_"), ("E_mean_rate_hz", "E_rate_hz_"),
                               ("I_mean_rate_hz", "I_rate_hz_")):
            summary.update(distribution([r[metric] for r in good], prefix))
        summary.update(silent_fraction_mean=float(np.mean([r["silent_fraction"] for r in good])) if good else None,
                       silent_fraction_median=float(np.median([r["silent_fraction"] for r in good])) if good else None,
                       trials_over_80pct_silent_fraction=float(np.mean([r["silent_fraction"] > .8 for r in good])) if good else None,
                       max_neuron_rate_median_hz=float(np.median([r["max_neuron_rate_hz"] for r in good])) if good else None,
                       max_neuron_rate_overall_hz=max((r["max_neuron_rate_hz"] for r in good), default=None))
        isi = pooled_isi[(base["input_g_us"], base["condition"])]
        summary.update(isi_count=int(isi.size), min_isi_ms=float(isi.min()) if isi.size else None,
                       median_isi_ms=float(np.median(isi)) if isi.size else None,
                       isi_below_2ms_fraction=float(np.mean(isi < 2.)) if isi.size else None)
        if MODEL_NAME == "AdEx":
            summary["depolarisation_pattern"] = "Not classified as block: AdEx has no sodium inactivation"
        else:
            summary["depolarisation_pattern"] = "Inspect voltage and rate distributions; no biological block classification"
        summaries.append(summary)
        for digit in range(10):
            selected = [r for r in group if r["label"] == digit]
            assert len(selected) == 5
            valid = [r for r in selected if r["valid"]]
            item = {"model": MODEL_NAME, "condition": base["condition"], "input_g_us": base["input_g_us"],
                    "digit": digit, "n_trials": len(selected), "valid_trials": len(valid), "valid": len(valid) == 5,
                    **distribution([r["mean_rate_hz"] for r in valid], "rate_hz_")}
            for key in ("silent_fraction", "E_mean_rate_hz", "I_mean_rate_hz"):
                item[key] = float(np.mean([r[key] for r in valid])) if valid else None
            digits.append(item)
    return summaries, digits


def plot_posthoc(outdir, rows, values, conditions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"FF": "#2878b5", "REC": "#cf6327"}

    def points(ax, groups, labels, colour):
        for k, data in enumerate(groups):
            a = np.asarray(data, dtype=float)
            if a.size:
                jitter = .16 * np.sin(np.arange(a.size) * 2.39996323)
                ax.scatter(k + jitter, a, s=16, alpha=.65, color=colour, zorder=3)
                q1, med, q3 = np.percentile(a, [25, 50, 75])
                ax.plot([k, k], [q1, q3], color="black", lw=4, zorder=4)
                ax.plot([k-.12, k+.12], [med, med], color="black", lw=2, zorder=5)
        ax.set_xticks(range(len(labels)), labels)
        ax.grid(axis="y", alpha=.2)

    for metric, filename, ylabel in (("mean_rate_hz", "posthoc_firing_rate_distribution.png", "Trial population rate (Hz)"),
                                    ("silent_fraction", "posthoc_silent_fraction_distribution.png", "Trial silent-neuron fraction")):
        fig, axes = plt.subplots(1, len(conditions), figsize=(6*len(conditions), 4), squeeze=False, sharey=True)
        for ax, condition in zip(axes.flat, conditions):
            points(ax, [[r[metric] for r in rows if r["valid"] and r["condition"] == condition and r["input_g_us"] == g]
                        for g in values], [f"{g:g}" for g in values], colours[condition])
            ax.set(title=condition, xlabel="Input conductance per synapse (uS)", ylabel=ylabel)
            if metric == "silent_fraction": ax.set_ylim(-.03, 1.03)
        fig.suptitle(f"{MODEL_NAME}: 50 balanced trials | dots = trials; black = median and IQR")
        fig.tight_layout(); fig.savefig(outdir / filename, dpi=170); plt.close(fig)

    fig, axes = plt.subplots(len(conditions), 2, figsize=(12, 4*len(conditions)), squeeze=False, sharey=True)
    for ci, condition in enumerate(conditions):
        for pi, pop in enumerate(("E", "I")):
            ax = axes[ci, pi]
            points(ax, [[r[pop+"_mean_rate_hz"] for r in rows if r["valid"] and r["condition"] == condition and r["input_g_us"] == g]
                        for g in values], [f"{g:g}" for g in values], colours[condition])
            ax.set(title=f"{condition} / {pop}", xlabel="Input conductance (uS)", ylabel="Trial population rate (Hz)")
    fig.suptitle(f"{MODEL_NAME}: E/I distributions | dots = trials; black = median and IQR")
    fig.tight_layout(); fig.savefig(outdir / "posthoc_EI_rates.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(len(conditions), len(values), figsize=(5*len(values), 3.7*len(conditions)), squeeze=False, sharey="col")
    for ci, condition in enumerate(conditions):
        for gi, g in enumerate(values):
            ax = axes[ci, gi]
            points(ax, [[r["mean_rate_hz"] for r in rows if r["valid"] and r["condition"] == condition and r["input_g_us"] == g and r["label"] == digit]
                        for digit in range(10)], [str(d) for d in range(10)], colours[condition])
            ax.set(title=f"{condition}, g={g:g} uS", xlabel="Original English digit", ylabel="Trial population rate (Hz)")
    fig.suptitle(f"{MODEL_NAME}: five trials per digit | black = median and IQR | rate scale differs by conductance")
    fig.tight_layout(); fig.savefig(outdir / "posthoc_digit_rates.png", dpi=170); plt.close(fig)


def compare_posthoc_directories(outdir, otherdir):
    """Explicitly requested counterpart; never guesses a run or alters defaults."""
    otherdir = Path(otherdir).expanduser().resolve()
    directories = [otherdir, outdir]
    metas = [json.loads((d / "metadata.json").read_text(encoding="utf-8")) for d in directories]
    assert {m["model"] for m in metas} == {"AdEx", "HH"}
    assert all(m["run_type"] == "input_g_posthoc" for m in metas)
    with np.load(otherdir / "posthoc_shared.npz", allow_pickle=False) as a, np.load(outdir / "posthoc_shared.npz", allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key], err_msg="Cross-model " + key)
        arrays_checked = list(a.files)
    # All common non-model protocol fields must agree, not merely the random arrays.
    exempt = {"input_g_values", "compare_posthoc_dir", "output_dir", "threshold_mv"}
    common = metas[0]["config"].keys() & metas[1]["config"].keys()
    checked = sorted(common - exempt)
    for key in checked:
        assert metas[0]["config"][key] == metas[1]["config"][key], key
    for key in ("seeds", "geometry", "synapse_parameters_at_baseline"):
        assert metas[0][key] == metas[1][key], key
    report = {"status": "passed", "directories": [str(d) for d in directories],
              "models": [m["model"] for m in metas], "shared_arrays_exact": arrays_checked,
              "matched_config_fields": checked, "seeds_geometry_synapses_equal": True,
              "trial_ids": metas[0]["trial_ids"], "digit_counts": metas[0]["validation"]["posthoc_selection"]["digit_counts"],
              "numerical_failures": {m["model"]: m["failures"] for m in metas},
              "note": "Same inputs and initial voltages; neuron-specific gates/states differ by definition. Equal input_g is not functional equivalence."}
    rows = []
    for d in directories:
        with (d / "posthoc_input_g_summary.csv").open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    write_csv_checked(outdir / "posthoc_cross_model_comparison.csv", rows)
    write_json(outdir / "posthoc_cross_model_validation.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for mi, model in enumerate(("AdEx", "HH")):
        for ci, condition in enumerate(("FF", "REC")):
            ax = axes[mi, ci]
            subset = sorted([r for r in rows if r["model"] == model and r["condition"] == condition], key=lambda r: float(r["input_g_us"]))
            for k, r in enumerate(subset):
                if not r["rate_hz_median"]: continue
                lo, q1, med, q3, hi, mean = [float(r["rate_hz_"+x]) for x in ("p05", "p25", "median", "p75", "p95", "mean")]
                ax.plot([k,k], [lo,hi], color="#777777", lw=2)
                ax.plot([k,k], [q1,q3], color="#2878b5", lw=8)
                ax.scatter(k, med, marker="_", color="black", s=180, zorder=4)
                ax.scatter(k, mean, marker="D", color="#cf6327", s=30, zorder=4)
            ax.set_xticks(range(len(subset)), [f"{float(r['input_g_us']):g}" for r in subset])
            ax.set(title=f"{model} / {condition}", xlabel="Model-specific input conductance (uS)", ylabel="Trial population rate (Hz)")
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Same 50 trials | grey: 5–95%; blue: IQR; black: median; orange: mean\nModel panels use separate rate scales; conductance is not functional equivalence")
    fig.tight_layout(); fig.savefig(outdir / "posthoc_cross_model_comparison.png", dpi=170); plt.close(fig)
    return report


def sweep_values(c):
    values = [float(x.strip()) for x in c.input_g_values.split(",")]
    if not values or not all(np.isfinite(x) and x > 0 for x in values):
        raise ValueError("--input-g-values must contain finite positive conductances in uS")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate sweep conductances are not allowed")
    return values


def array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def recording_hash(record):
    return [array_hash(x) for x in record]


def write_csv_checked(path, rows):
    """Checkpoint completed measurements and verify the CSV's exact text values."""
    if not rows:
        return
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    with path.open(newline="", encoding="utf-8") as fh:
        loaded = list(csv.DictReader(fh))
    assert len(loaded) == len(rows)
    for actual, expected in zip(loaded, rows):
        assert actual == {key: "" if value is None else str(value) for key, value in expected.items()}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def initial_channel_states(model):
    nodes = model["net"].nodes.iloc[N_INPUT:]
    return {key: nodes[key].to_numpy(dtype=float).tolist() for key in nodes.columns
            if key.startswith(MODEL_NAME + "_") and key not in model["neuron_params"]}


def verify_input_parameter_isolation(model, input_g):
    """Compare the actual full Jaxley parameter trees, including recurrent types."""
    net = model["net"]
    net.to_jax()
    baseline = net.get_all_parameters([])
    changed = net.get_all_parameters(model["parameter_state"](input_g))
    assert baseline.keys() == changed.keys()
    for key in baseline:
        if key == "InputExc_gS":
            np.testing.assert_array_equal(np.asarray(changed[key]), input_g)
        else:
            for a, b in zip(jax.tree_util.tree_leaves(baseline[key]),
                            jax.tree_util.tree_leaves(changed[key])):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=key)
    return True


SWEEP_MEASUREMENTS = (
    "mean_rate_hz", "slowest_cell_rate_hz", "median_voltage_mv", "E_mean_rate_hz",
    "I_mean_rate_hz", "silent_fraction", "E_silent_fraction", "I_silent_fraction",
    "post_mean_rate_hz", "post_E_mean_rate_hz", "post_I_mean_rate_hz",
    "pre_spikes", "pre_mean_rate_hz", "settling_voltage_drift_mv",
    "depolarised_sample_fraction", "voltage_min_mv", "voltage_max_mv")


def sweep_measurement(record, c):
    v, flags, _ = record
    onset, offset = c.steps(c.settle_ms), c.steps(c.settle_ms + c.input_ms)
    input_v = v[:, onset:offset]
    rates = flags[:, onset:offset].sum(axis=1) / (c.input_ms / 1000)
    post_rates = flags[:, offset:].sum(axis=1) / (c.post_ms / 1000) if c.post_ms else None
    pre_spikes = int(flags[:, :onset].sum())
    tail = max(0, onset - c.steps(min(10., c.settle_ms)))
    row = dict(mean_rate_hz=float(rates.mean()), slowest_cell_rate_hz=float(rates.min()),
               median_voltage_mv=float(np.median(input_v)),
               E_mean_rate_hz=float(rates[:c.n_exc].mean()), I_mean_rate_hz=float(rates[c.n_exc:].mean()),
               silent_fraction=float(np.mean(rates == 0)),
               E_silent_fraction=float(np.mean(rates[:c.n_exc] == 0)),
               I_silent_fraction=float(np.mean(rates[c.n_exc:] == 0)),
               post_mean_rate_hz=float(post_rates.mean()) if post_rates is not None else None,
               post_E_mean_rate_hz=float(post_rates[:c.n_exc].mean()) if post_rates is not None else None,
               post_I_mean_rate_hz=float(post_rates[c.n_exc:].mean()) if post_rates is not None else None,
               pre_spikes=pre_spikes, pre_mean_rate_hz=pre_spikes / c.n_out / (c.settle_ms / 1000),
               settling_voltage_drift_mv=float(np.max(np.abs(v[:, onset] - v[:, tail]))),
               depolarised_sample_fraction=float(np.mean(input_v > -40.)),
               voltage_min_mv=float(v.min()), voltage_max_mv=float(v.max()))
    assert set(row) == set(SWEEP_MEASUREMENTS)
    return row, rates, input_v


def sweep_shared_arrays(c, trial_ids, labels, speakers, inputs, recurrent, initial, models):
    arrays = {"neuron_ids": np.arange(c.n_out, dtype=np.int32),
              "population": np.array(["E"] * c.n_exc + ["I"] * c.n_inh),
              "input_channel_ids": np.arange(N_INPUT, dtype=np.int32),
              "trial_ids": trial_ids, "labels": labels, "speaker_ids": speakers,
              "source_split": np.full(len(trial_ids), c.split), "trial_order": np.arange(len(trial_ids)),
              "input_connectivity": inputs, "recurrent_connectivity": recurrent,
              "initial_voltage_mv": initial,
              "receptive_field_centres": np.r_[np.linspace(0, 699, c.n_exc), np.linspace(0, 699, c.n_inh)]}
    for name, model in models.items():
        for key, value in edge_arrays(model).items():
            arrays[f"{name}_edges_{key}"] = value
    return arrays


def make_sweep_metadata(c, models, seeds, data_path, shared, provenance, compatibility, validation):
    exemplar = next(iter(models.values()))
    return {
        "schema_version": 1, "run_type": "input_g_sweep", "model": MODEL_NAME,
        "status": "running", "config": asdict(c), "seeds": seeds,
        "conditions": ["FF", "REC"] if c.conditions == "both" else [c.conditions],
        "input_g_values_us": sweep_values(c), "ordinary_baseline_input_g_us": c.input_g,
        "notebook_reference_input_g_us": 3e-4,
        "notebook_reference": "https://github.com/CNNC-Lab/compbio-2026/blob/cf69a137c3709913031f09598d494846d5272393/notebooks/08_jaxley_hh_layer.ipynb",
        "input_g_equivalence": "Notebook G_SYN is input_g, the maximal conductance of each BC input synapse in uS",
        "source_hdf5": str(data_path), "source_hdf5_size_bytes": data_path.stat().st_size,
        "source_hdf5_mtime_ns": data_path.stat().st_mtime_ns,
        "trial_ids": shared["trial_ids"].tolist(), "labels": shared["labels"].tolist(),
        "speaker_ids": shared["speaker_ids"].tolist(), "split": c.split,
        "hashes": {key: array_hash(shared[key]) for key in
                   ("trial_ids", "labels", "speaker_ids", "input_connectivity", "recurrent_connectivity", "initial_voltage_mv")},
        "stimulus_sha256": [], "stimulus_hash_definition": "SHA256 of contiguous float64 clamp bytes, identical to ordinary archives",
        "input_wiring_implementation": provenance,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_versions": {key: importlib.metadata.version(key) for key in
                             ("jax", "jaxlib", "jaxley", "numpy", "tables", "matplotlib", "pandas")},
        "python": platform.python_version(), "jax_devices": [str(x) for x in jax.devices()],
        "jax_enable_x64": True, "compatibility": compatibility,
        "neuron_parameters": exemplar["neuron_params"],
        "initial_neuronal_states": initial_channel_states(exemplar),
        "geometry": {"radius_um": 10., "length_um": 10., "capacitance_uf_cm2": 1.,
                     "axial_resistivity_ohm_cm": float(exemplar["net"].nodes.axial_resistivity.iloc[0])},
        "synapse_parameters_at_baseline": {name: x["synapse_params"] for name, x in models.items()},
        "units": {"input_g_and_recurrent_g": "uS per synapse", "intrinsic_conductances": "S/cm2",
                  "voltage": "mV", "time": "ms", "rates": "Hz", "synapse_k_minus": "1/ms"},
        "initial_synapse_s": 0., "initial_input_leak_g": 0.,
        "reset_protocol": "Fresh defined voltage, channel states and zero synaptic gates on every integrate call, then identical settling",
        "replay_mapping": "float64(seconds)*1000; ceil(t/dt); [0,input_ms); discard ceil at offset; clamp index settle+bin-1; clipped OR pulses",
        "replay_lag": "Clamp is applied at step end; synapse observes it next step; retain one dt numerical lag",
        "aggregation": {
            "primary_window": "[0,input_ms), integer sample indices settle_steps:input_offset_steps",
            "post_window": "[input_ms,input_ms+post_ms], final endpoint included; rate undefined if post_ms=0",
            "pre_window": "[-settle_ms,0)",
            "mean_rate_hz": "Mean over neurons and selected trials of input-window count divided by input duration",
            "slowest_cell_rate_hz": "Minimum across neurons after averaging each neuron's rate across trials",
            "median_voltage_mv": "Exact median pooled over ALL selected trials, neurons and input-window voltage samples",
            "silent_fraction": "Fraction of neurons with zero input-window spikes over all selected trials; per-trial fractions in trial CSV",
            "elapsed_seconds": "Simulation call only, including first-call JIT compilation; excludes validation and diagnostics",
            "trial_rate_sd_hz": "Sample SD of per-trial population mean rates; NOT uncertainty across independent networks",
            "depolarisation_pattern": "Flag if median V>-40 mV and mean rate falls below 80% of the maximum at lower valid g; suggestive only",
            "failed_points": "Failed trials have blank metrics, not zeros; aggregates are blank unless every selected trial is valid"},
        "waveform_comparison_note": "Equal maximal recurrent g across models does not imply equal effective drive: activation depends on the actual presynaptic voltage waveform",
        "calibration": "None. No best conductance chosen; ordinary defaults unchanged",
        "validation": validation, "per_trial_diagnostics": [], "replay_per_trial": [],
        "point_results": [], "failures": []}


def plot_sweep(outdir, summaries, c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for condition, colour, marker in (("FF", "#2878b5", "o"), ("REC", "#cf6327", "s")):
        rows = sorted([x for x in summaries if x["condition"] == condition], key=lambda x: x["input_g_us"])
        if not rows:
            continue
        g = [x["input_g_us"] for x in rows]
        values = lambda key: [np.nan if x[key] is None else x[key] for x in rows]
        axes[0, 0].plot(g, values("mean_rate_hz"), marker=marker, color=colour, label=condition)
        axes[0, 1].plot(g, values("median_voltage_mv"), marker=marker, color=colour, label=condition)
        for pop, style in (("E", "-"), ("I", "--")):
            axes[1, 0].plot(g, values(pop + "_mean_rate_hz"), style, marker=marker, color=colour, label=f"{condition} {pop}")
        axes[1, 1].plot(g, values("silent_fraction"), marker=marker, color=colour, label=condition)
    for ax, label in zip(axes.flat, ("Mean output rate (Hz)", "Pooled median voltage (mV)",
                                   "E / I mean rate (Hz)", "Silent fraction across trials")):
        ax.set_xscale("log")
        ax.set(xlabel="Input conductance per synapse (uS)", ylabel=label)
        ax.axvline(c.input_g, color="#555555", ls=":", lw=1, label="Ordinary baseline")
        ax.axvline(3e-4, color="#888888", ls="-.", lw=1, label="Notebook HH reference")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    axes[1, 1].set_ylim(-.03, 1.03)
    fig.suptitle(f"{MODEL_NAME}: input conductance diagnostic | input window only | fixed network")
    fig.tight_layout()
    fig.savefig(outdir / "input_g_sweep.png", dpi=170)
    plt.close(fig)


def run_input_g_sweep(c, models, inputs, recurrent, initial, seeds, data_path,
                      provenance, compatibility, validation, conversion):
    values = sweep_values(c)
    prefix = "posthoc_input_g" if c.posthoc_input_g else "input_g_sweep"
    shared_name = "posthoc_shared" if c.posthoc_input_g else "sweep_shared"
    rates_name = "posthoc_neuron_rates" if c.posthoc_input_g else "sweep_neuron_rates"
    pooled_isi = {}
    conditions = ["FF", "REC"] if c.conditions == "both" else [c.conditions]
    outdir = Path(c.output_dir).expanduser().resolve() / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") +
        f"_{MODEL_NAME}_input_g_{'posthoc' if c.posthoc_input_g else 'sweep'}_" + uuid.uuid4().hex[:8])
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"Sweep directory: {outdir}", flush=True)
    with tables.open_file(data_path, mode="r") as fh:
        trial_ids, labels, speakers = select_trials(fh, c, seeds["selection"])
        if c.posthoc_input_g:
            validation["posthoc_selection"] = validate_posthoc_selection(c, trial_ids, labels)
        shared = sweep_shared_arrays(c, trial_ids, labels, speakers, inputs, recurrent, initial, models)
        metadata = make_sweep_metadata(c, models, seeds, data_path, shared, provenance, compatibility, validation)
        if c.posthoc_input_g:
            metadata["run_type"] = "input_g_posthoc"
            metadata["calibration"] = "Post-hoc operating-point evaluation; no decoder accuracy optimisation, no automatic baseline update"
            metadata["aggregation"].update(
                posthoc_distributions="Empirical distributions of trial means; sample SD ddof=1, numpy linear percentiles; IQR=p75-p25. Available valid trials only; counts and validity explicit.",
                posthoc_silence="silent_fraction_mean/median aggregate per-trial fractions; trials_over_80pct_silent_fraction uses strict >0.8. Legacy silent_fraction is silence across all trials.",
                posthoc_voltage="Per-trial voltage extrema and medians use input window only; summary median_voltage_mv is exact pooled sample median, not median of medians.",
                posthoc_ISI="Within each neuron and trial, between successive input-window spikes only; no cross-trial/settling/post intervals. Pooled intervals for aggregate median/fraction; undefined without intervals.",
                posthoc_maximum="max_neuron_rate_hz is maximum within each trial; summary reports its median and overall maximum.",
                posthoc_failure="Trial metrics blank and per-neuron rates NaN when invalid; distributions label valid counts. Pooled voltage/legacy aggregates blank if any trial invalid.",
                depolarisation_pattern="HH high voltage with reduced firing warrants inspection only; AdEx is never classified as sodium-inactivation block.")
        metadata["spike_detection"] = {"method": "AdEx_spikes at recorded sample k",
                                        "timestamp_ms": "(k-settle_steps)*dt", "diagnostic_threshold_mv": c.threshold_mv}
        write_json(outdir / "metadata.json", metadata)
        rows, summaries, first_trial_hashes = [], [], {}
        neuron_rates = np.full((len(values), len(conditions), len(trial_ids), c.n_out), np.nan)
        valid_trials = np.zeros(neuron_rates.shape[:3], dtype=bool)
        chosen_g = c.input_g if c.input_g in values else values[0]
        started = time.perf_counter()
        for point, g in enumerate(values):
            print(f"{MODEL_NAME} sweep {point+1}/{len(values)}: input_g={g:g} uS; {len(trial_ids)} trials", flush=True)
            for model in models.values():
                verify_input_parameter_isolation(model, g)
            validation.setdefault("only_input_g_changes", []).append(g)
            event_trials = {name: [] for name in conditions}
            pre_trials = {name: [] for name in conditions}
            point_dir = outdir / f"g_{point:03d}_{g:.8g}"
            if c.save_sweep_events or c.diagnostic_trials:
                point_dir.mkdir(exist_ok=False)
            with ExitStack() as stack:
                scratch = {name: stack.enter_context(tempfile.TemporaryFile(dir=outdir)) for name in conditions}
                point_rows = {name: [] for name in conditions}
                point_isi = {name: [] for name in conditions}
                for order, (trial, label, speaker) in enumerate(zip(trial_ids, labels, speakers)):
                    waveform, losses, raw, pulses = replay(fh.root.spikes.times[int(trial)], fh.root.spikes.units[int(trial)], c)
                    digest = hashlib.sha256(waveform.tobytes()).hexdigest()
                    if point == 0:
                        metadata["stimulus_sha256"].append(digest)
                        metadata["replay_per_trial"].append(losses)
                    else:
                        assert digest == metadata["stimulus_sha256"][order], "Stimulus changed across sweep"
                    recordings = {}
                    for ci, name in enumerate(conditions):
                        row = {"model": MODEL_NAME, "condition": name, "input_g_us": g,
                               "trial_order": order, "trial_id": int(trial), "label": int(label),
                               "speaker_id": int(speaker), "split": c.split, "valid": False,
                               "error": "", "elapsed_seconds": 0., **dict.fromkeys(SWEEP_MEASUREMENTS)}
                        if c.posthoc_input_g:
                            row.update(dict.fromkeys(POSTHOC_EXTRA))
                        t0 = time.perf_counter()
                        try:
                            result = simulate(models[name], waveform, c, input_g=g)
                        except FloatingPointError as error:
                            # Only recognised numerical validity errors are recoverable.
                            # Assertion/shape/API/programming errors deliberately propagate.
                            row["error"] = str(error)
                            row["elapsed_seconds"] = time.perf_counter() - t0
                            metadata["failures"].append({"condition": name, "input_g_us": g,
                                                         "trial_id": int(trial), "error": str(error)})
                            print(f"  {name} trial {trial}: NUMERICAL FAILURE: {error}", flush=True)
                        else:
                            row["elapsed_seconds"] = time.perf_counter() - t0
                            measurements, rates, input_v = sweep_measurement(result, c)
                            row.update(measurements, valid=True)
                            if c.posthoc_input_g:
                                extra, isi = posthoc_measurement(result, rates, input_v, c)
                                row.update(extra)
                                point_isi[name].append(isi)
                            neuron_rates[point, ci, order] = rates
                            valid_trials[point, ci, order] = True
                            np.ascontiguousarray(input_v).tofile(scratch[name])
                            recordings[name] = result
                            metadata["per_trial_diagnostics"].append({"condition": name, "input_g_us": g,
                                "trial_id": int(trial), "diagnostics": diagnostics(*result, c)})
                            if c.save_sweep_events:
                                event, pre = extract_events(result[1], c)
                                event_trials[name].append(event)
                                pre_trials[name].append(pre)
                            if order == 0:
                                first_trial_hashes[(point, name)] = recording_hash(result)
                                if c.validate and point == 0:
                                    repeat = simulate(models[name], waveform, c, input_g=g)
                                    validation[f"{name}_immediate_repeat"] = compare_runs(result, repeat, "sweep immediate repeat", atol=0.)
                                if c.validate and g == chosen_g:
                                    ordinary_config = replace(c, input_g=g, sweep_input_g=False)
                                    ordinary = build_network(ordinary_config, inputs, recurrent, initial, name, conversion)
                                    ordinary_result = simulate(ordinary, waveform, ordinary_config)
                                    validation[f"{name}_ordinary_matches_sweep_first_trial"] = {
                                        "input_g_us": g, **compare_runs(result, ordinary_result, "ordinary versus sweep")}
                                    del ordinary, ordinary_result
                            print(f"  {name} trial {trial}: {row['mean_rate_hz']:.3f} Hz; "
                                  f"median V {row['median_voltage_mv']:.2f} mV", flush=True)
                        rows.append(row)
                        point_rows[name].append(row)
                        write_csv_checked(outdir / f"{prefix}_trials.csv", rows)
                        write_json(outdir / "metadata.json", metadata)
                    if c.validate and point == 0 and order == 0:
                        try:
                            ff = recordings.get("FF")
                            if ff is None:
                                ff = simulate(models["FF"], waveform, c, input_g=g)
                            zero = simulate(models["REC"], waveform, c, recurrent_scale=0., input_g=g)
                            validation["zero_recurrence_matches_FF"] = compare_runs(ff, zero, "sweep zero recurrence")
                        except FloatingPointError as error:
                            validation["zero_recurrence_matches_FF"] = {"passed": False, "numerical_error": str(error)}
                    if order < c.diagnostic_trials and recordings:
                        plot_trial(point_dir / f"trial_{order:03d}_source_{trial}", raw, pulses, recordings, c,
                                   f"{MODEL_NAME}, input g={g:g} uS, {c.split} trial {trial}, label {label}")
                for ci, name in enumerate(conditions):
                    valid = bool(valid_trials[point, ci].all())
                    summary = {"model": MODEL_NAME, "condition": name, "input_g_us": g,
                               "valid": valid, "selected_trials": len(trial_ids),
                               "valid_trials": int(valid_trials[point, ci].sum()),
                               **dict.fromkeys(SWEEP_MEASUREMENTS), "trial_rate_sd_hz": None,
                               "elapsed_seconds": sum(x["elapsed_seconds"] for x in point_rows[name]),
                               "depolarisation_pattern": "not assessed"}
                    if valid:
                        rates = neuron_rates[point, ci]
                        by_neuron = rates.mean(axis=0)
                        for key in SWEEP_MEASUREMENTS:
                            vals = [x[key] for x in point_rows[name]]
                            summary[key] = float(np.mean(vals)) if all(x is not None for x in vals) else None
                        summary["slowest_cell_rate_hz"] = float(by_neuron.min())
                        for pop, region in (("", slice(None)), ("E_", slice(0, c.n_exc)), ("I_", slice(c.n_exc, c.n_out))):
                            summary[pop + "silent_fraction"] = float(np.mean(by_neuron[region] == 0))
                        summary["trial_rate_sd_hz"] = float(rates.mean(axis=1).std(ddof=1)) if len(trial_ids) > 1 else None
                        summary["pre_spikes"] = sum(x["pre_spikes"] for x in point_rows[name])
                        summary["settling_voltage_drift_mv"] = max(x["settling_voltage_drift_mv"] for x in point_rows[name])
                        summary["voltage_min_mv"] = min(x["voltage_min_mv"] for x in point_rows[name])
                        summary["voltage_max_mv"] = max(x["voltage_max_mv"] for x in point_rows[name])
                        scratch[name].flush()
                        size = scratch[name].tell() // np.dtype(np.float64).itemsize
                        expected = len(trial_ids) * c.n_out * c.steps(c.input_ms)
                        assert size == expected
                        mapped = np.memmap(scratch[name], dtype=np.float64, mode="r+", shape=(size,))
                        summary["median_voltage_mv"] = float(np.median(mapped, overwrite_input=True))
                        del mapped
                    summaries.append(summary)
                    if c.posthoc_input_g:
                        pooled_isi[(g, name)] = np.concatenate(point_isi[name]) if point_isi[name] else np.empty(0)
            # Recompute this annotation using numerical g order, not evaluation order.
            for summary in summaries:
                if summary["valid"]:
                    lower = [x["mean_rate_hz"] for x in summaries if x["valid"] and
                             x["condition"] == summary["condition"] and x["input_g_us"] < summary["input_g_us"]]
                    pattern = lower and summary["median_voltage_mv"] > -40 and summary["mean_rate_hz"] < .8 * max(lower)
                    summary["depolarisation_pattern"] = "high voltage with reduced spiking; inspect for possible block" if pattern else "not flagged"
            metadata["point_results"] = summaries
            write_csv_checked(outdir / f"{prefix}_summary.csv", summaries)
            if c.save_sweep_events:
                if all(x["valid"] for name in conditions for x in point_rows[name]):
                    point_arrays = {**shared, "stimulus_sha256": np.asarray(metadata["stimulus_sha256"])}
                    for name in conditions:
                        for suffix, trials in (("", event_trials[name]), ("_pre", pre_trials[name])):
                            t, u, offsets = pack_events(trials)
                            point_arrays.update({f"{name}{suffix}_times_ms": t,
                                                 f"{name}{suffix}_neuron_ids": u, f"{name}{suffix}_offsets": offsets})
                    point_meta = json.loads(json.dumps(metadata))
                    point_meta.update(run_type="sweep_event_archive", status="valid", input_g_us=g)
                    point_meta["config"]["input_g"] = g
                    point_meta["synapse_parameters"] = json.loads(json.dumps(point_meta["synapse_parameters_at_baseline"]))
                    for parameters in point_meta["synapse_parameters"].values():
                        parameters["InputExc"]["InputExc_gS"] = g
                    save_results(point_dir, point_arrays, point_meta, replace(c, input_g=g))
                else:
                    write_json(point_dir / "FAILED.json", {"reason": "Incomplete point; no empty events substituted for failed trials"})
            write_json(outdir / "metadata.json", metadata)
        if c.validate:
            # Trial A is revisited after other trials and conductances, in reverse g order.
            waveform, _, _, _ = replay(fh.root.spikes.times[int(trial_ids[0])], fh.root.spikes.units[int(trial_ids[0])], c)
            reverse_checks = []
            for point in reversed(range(len(values))):
                for name in conditions:
                    if (point, name) in first_trial_hashes:
                        result = simulate(models[name], waveform, c, input_g=values[point])
                        assert recording_hash(result) == first_trial_hashes[(point, name)], "Order/reset changed output"
                        reverse_checks.append({"condition": name, "input_g_us": values[point], "exact": True})
            validation["reverse_order_and_reset_first_trial"] = reverse_checks
            validation["intervening_distinct_trials"] = len(trial_ids) - 1
        shared["stimulus_sha256"] = np.asarray(metadata["stimulus_sha256"])
        np.savez_compressed(outdir / f"{shared_name}.npz", **shared)
        with np.load(outdir / f"{shared_name}.npz", allow_pickle=False) as loaded:
            for key in shared:
                np.testing.assert_array_equal(shared[key], loaded[key])
        np.savez_compressed(outdir / f"{rates_name}.npz", input_g_us=values,
                            conditions=np.asarray(conditions), trial_ids=trial_ids,
                            neuron_ids=np.arange(c.n_out), rates_hz=neuron_rates, valid=valid_trials)
        with np.load(outdir / f"{rates_name}.npz", allow_pickle=False) as loaded:
            np.testing.assert_array_equal(loaded["rates_hz"], neuron_rates)
            np.testing.assert_array_equal(loaded["valid"], valid_trials)
        validation["csv_and_npz_reload"] = True
        metadata["status"] = "partially_valid" if metadata["failures"] else "complete"
        metadata["elapsed_seconds"] = time.perf_counter() - started
        write_json(outdir / "metadata.json", metadata)
        if c.posthoc_input_g:
            posthoc_summary, digit_rows = posthoc_tables(rows, summaries, pooled_isi)
            write_csv_checked(outdir / "posthoc_input_g_summary.csv", posthoc_summary)
            write_csv_checked(outdir / "posthoc_input_g_by_digit.csv", digit_rows)
            plot_posthoc(outdir, rows, values, conditions)
            metadata["point_results"] = posthoc_summary
            metadata["validation"]["posthoc_table_dimensions"] = {
                "trial_rows": len(rows), "summary_rows": len(posthoc_summary), "digit_rows": len(digit_rows)}
            assert len(rows) == len(values) * len(conditions) * 50
            assert len(digit_rows) == len(values) * len(conditions) * 10
            metadata["elapsed_seconds"] = time.perf_counter() - started
            write_json(outdir / "metadata.json", metadata)
            if c.compare_posthoc_dir:
                metadata["validation"]["cross_model"] = compare_posthoc_directories(outdir, c.compare_posthoc_dir)
                write_json(outdir / "metadata.json", metadata)
        else:
            plot_sweep(outdir, summaries, c)
        marker = "PARTIAL.txt" if metadata["failures"] else "COMPLETE.txt"
        (outdir / marker).write_text(f"{MODEL_NAME} diagnostic sweep: {metadata['status']}\n", encoding="utf-8")
        print(f"{MODEL_NAME} sweep {metadata['status']}: {outdir}", flush=True)


def main():
    c = parse_config()
    started = time.perf_counter()
    seeds = seeds_for(c)
    data_path = locate_data(c)
    conversion, compatibility = adex_compatibility()
    print(f"Jaxley {importlib.metadata.version('jaxley')}; AdEx internal current conversion={conversion:g}", flush=True)
    primitive_checks = validate_primitives(c, conversion)
    inputs, recurrent, wiring_provenance = connectivity(c, seeds)
    initial = np.random.default_rng(seeds["initial"]).normal(c.initial_mv, c.initial_sd_mv, c.n_out)
    if np.any(initial >= c.threshold_mv):
        raise ValueError("Initial voltages must be below the diagnostic spike threshold")
    conditions = ["FF", "REC"] if c.conditions == "both" else [c.conditions]
    needed = ["FF", "REC"] if c.validate else conditions
    models = {}
    for name in needed:
        print(f"Building {name}: {c.n_exc} E + {c.n_inh} I, {int(inputs.sum())} input edges", flush=True)
        models[name] = build_network(c, inputs, recurrent, initial, name, conversion)
    validation = {**primitive_checks, **validate_structure(models, inputs, recurrent, initial, c)}
    if c.sweep_input_g or c.posthoc_input_g:
        run_input_g_sweep(c, models, inputs, recurrent, initial, seeds, data_path,
                          wiring_provenance, compatibility, validation, conversion)
        return
    output_root = Path(c.output_dir).expanduser().resolve()
    outdir = output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {outdir}", flush=True)
    print(f"Recurrent edge counts: {validation['recurrent_counts']}", flush=True)
    outputs = {name: [] for name in conditions}
    pre_outputs = {name: [] for name in conditions}
    replay_reports, trial_diagnostics, trial_hashes = [], [], []
    first_reference = None
    with tables.open_file(data_path, mode="r") as fh:
        trial_ids, labels, speakers = select_trials(fh, c, seeds["selection"])
        print(f"Selected {len(trial_ids)} {c.split} trials; original labels {np.unique(labels).tolist()}; "
              f"{c.settle_ms:g}+{c.input_ms:g}+{c.post_ms:g} ms, dt={c.dt_ms:g} ms", flush=True)
        for row, (trial, label) in enumerate(zip(trial_ids, labels)):
            wave, replay_report, raw, pulses = replay(fh.root.spikes.times[int(trial)],
                                                       fh.root.spikes.units[int(trial)], c)
            replay_reports.append(replay_report)
            trial_hashes.append(hashlib.sha256(wave.tobytes()).hexdigest())
            print(f"Trial {row+1}/{len(trial_ids)}: source={trial}, label={label}, "
                  f"events={replay_report['accepted_events']}, excluded={replay_report['excluded_input_window']}, "
                  f"merged={replay_report['same_channel_grid_collisions'] + replay_report['additional_overlapping_or_adjacent_pulses']}", flush=True)
            record = {}
            diag = {}
            # In single-condition runs, the auxiliary control must also see trial B
            # before the final A-B-A reset check can claim an intervening trial.
            run_names = needed if c.validate and row < 2 else conditions
            for name in run_names:
                t0 = time.perf_counter()
                record[name] = simulate(models[name], wave, c)
                diag[name] = diagnostics(*record[name], c)
                if name in conditions:
                    event, pre = extract_events(record[name][1], c)
                    outputs[name].append(event)
                    pre_outputs[name].append(pre)
                print(f"  {name}: {time.perf_counter()-t0:.1f}s; " + "; ".join(
                    f"{p} {diag[name][p]['observation']['mean_rate_hz']:.2f} Hz, "
                    f"silent {diag[name][p]['observation']['silent_fraction']:.0%}"
                    for p in ("E", "I")) + f"; pre spikes={diag[name]['settling']['pre_spikes']}", flush=True)
                for warning in diag[name]["warnings"]:
                    print(f"  Diagnostic warning: {name}: {warning}", flush=True)
            if c.validate and row == 0:
                zero = simulate(models["REC"], wave, c, recurrent_scale=0.)
                validation["zero_recurrence_matches_FF"] = compare_runs(record["FF"], zero, "REC g=0 versus FF")
                for name in needed:
                    repeat = simulate(models[name], wave, c)
                    validation[name + "_repeat_same_initial_state"] = compare_runs(record[name], repeat, name + " repeat", atol=0.)
                first_reference = {name: tuple(x.copy() for x in record[name]) for name in needed}
                print("  Controls passed: zero recurrence equals FF; both trial repeats reproduce exactly", flush=True)
            if row < c.diagnostic_trials:
                plot_trial(outdir / f"trial_{row:03d}_source_{trial}", raw, pulses,
                           {name: record[name] for name in conditions}, c,
                           f"SHD {c.split} trial {trial}, original label {label}; E blue / I orange")
            trial_diagnostics.append(diag)
        if c.validate:
            # Repeat trial A after trial B, using the very same compiled models.
            wave, _, _, _ = replay(fh.root.spikes.times[int(trial_ids[0])],
                                    fh.root.spikes.units[int(trial_ids[0])], c)
            for name in needed:
                repeat = simulate(models[name], wave, c)
                validation[name + "_reset_after_other_trials"] = compare_runs(first_reference[name], repeat, name + " reset after other trials", atol=0.)
            validation["reset_intervening_trial_count"] = len(trial_ids) - 1
    arrays = {"neuron_ids": np.arange(c.n_out, dtype=np.int32),
              "population": np.array(["E"] * c.n_exc + ["I"] * c.n_inh),
              "input_channel_ids": np.arange(N_INPUT, dtype=np.int32),
              "trial_ids": trial_ids, "labels": labels, "speaker_ids": speakers,
              "source_split": np.full(len(trial_ids), c.split),
              "trial_order": np.arange(len(trial_ids)),
              "input_connectivity": inputs, "recurrent_connectivity": recurrent,
              "initial_voltage_mv": initial,
              "receptive_field_centres": np.r_[np.linspace(0, 699, c.n_exc), np.linspace(0, 699, c.n_inh)],
              "stimulus_sha256": np.asarray(trial_hashes)}
    aggregate = {}
    for name in conditions:
        for suffix, trials in (("", outputs[name]), ("_pre", pre_outputs[name])):
            t, u, offsets = pack_events(trials)
            arrays.update({f"{name}{suffix}_times_ms": t, f"{name}{suffix}_neuron_ids": u,
                           f"{name}{suffix}_offsets": offsets})
        for key, value in edge_arrays(models[name]).items():
            arrays[f"{name}_edges_{key}"] = value
        counts = np.bincount(arrays[f"{name}_neuron_ids"], minlength=c.n_out)
        aggregate[name] = {}
        for pop, region in (("E", slice(0, c.n_exc)), ("I", slice(c.n_exc, c.n_out))):
            aggregate[name][pop] = {"mean_rate_hz": float(counts[region].mean() / len(trial_ids) / ((c.input_ms + c.post_ms) / 1000)),
                                    "silent_fraction_across_all_trials": float(np.mean(counts[region] == 0))}
    metadata = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(c), "conditions": conditions, "seeds": seeds,
        "source_hdf5": str(data_path), "source_hdf5_size_bytes": data_path.stat().st_size,
        "source_hdf5_mtime_ns": data_path.stat().st_mtime_ns,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "course_reference_commit": REFERENCE_COMMIT, "input_wiring_implementation": wiring_provenance,
        "package_versions": {name: importlib.metadata.version(name) for name in
                             ("jax", "jaxlib", "jaxley", "numpy", "tables", "matplotlib", "pandas")},
        "python": platform.python_version(), "platform": platform.platform(),
        "jax_devices": [str(d) for d in jax.devices()], "jax_enable_x64": True,
        "neuron_model": "Installed Jaxley AdEx with probed physical-current unit adapter",
        "compatibility": compatibility, "course_ADEX_PARAMS": ADEX_PARAMS,
        "neuron_parameters": models[needed[0]]["neuron_params"],
        "geometry": {"radius_um": 10., "length_um": 10., "capacitance_uf_cm2": 1.,
                     "axial_resistivity_ohm_cm": float(models[needed[0]]["net"].nodes.axial_resistivity.iloc[0]),
                     "physical_passive_tau_ms": 1 / (1000 * c.adex_g_l)},
        "synapse_parameters": {name: models[name]["synapse_params"] for name in needed},
        "parameter_units": {"AdEx_g_L_and_a": "S/cm2", "AdEx_b_and_w": "mA/cm2",
                            "gS": "uS", "k_minus": "1/ms (delta_t is ms)", "voltages": "mV"},
        "initial_states": {"input_v_mv": c.replay_rest_mv, "input_Leak_gLeak": 0.,
                           "input_Leak_eLeak_mv": float(models[needed[0]]["net"].nodes.Leak_eLeak.iloc[0]),
                           "output_v_mv": "initial_voltage_mv array", "AdEx_w": c.initial_w,
                           "AdEx_spikes": 0, "all_synapse_s": 0},
        "reset_protocol": "Every integrate call starts from model initial states, including settling; no final states reused",
        "replay_mapping": "float64 seconds*1000; ceil(t/dt); [0,input_ms); discard ceil==offset; clamp[settle_steps+bin-1]; OR pulses clipped at input offset",
        "replay_synapse_timing": "Clamp applied at end of step; synapses see it on next step; one dt numerical lag retained, not subtracted",
        "spike_detection": {"primary": "AdEx_spikes at sample k; t_ms=(k-settle_steps)*dt",
                            "diagnostic": f"upward voltage crossing of {c.threshold_mv} mV at sample k+1",
                            "reason": "Installed AdEx resets immediately; voltage crossings can miss reset events",
                            "observation_window_ms": [0., c.input_ms + c.post_ms], "right_endpoint_included": True},
        "connectivity_note": "Matrices [pre,post]; recurrent_connectivity is candidate REC wiring, actual per-condition edges also saved; global output IDs = 700+local ID",
        "tonotopy_note": "E and I centres each span 0..699; centres array is nominal and unused for random wiring",
        "calibration": "No automated parameter sweep. Input g=3e-5 uS from course AdEx example; recurrent strengths provisional. Unit correction is not parameter fitting.",
        "limitations": ["Small trial subsets do not validate biological firing regimes or generalisation",
                        "Installed AdEx has immediate reset, no explicit refractory period and no imposed AP waveform",
                        "Recurrent transmitter release is graded by AdEx upstroke voltage; its pulse shape differs from the clamped BC input",
                        "No explicit axonal delays; voltage-clamp replay has a retained one-step numerical lag",
                        "No decoding, training, or automatic conductance calibration"],
        "analysis_adapter": "Use offsets to form SHD times lists divided by 1000 (seconds), units lists, original labels and speaker_ids; supply n_channels=len(neuron_ids) explicitly",
        "replay_per_trial": replay_reports, "diagnostics_per_trial": trial_diagnostics,
        "aggregate": aggregate, "validation": validation,
        "elapsed_seconds_before_save": time.perf_counter() - started,
    }
    destination = save_results(outdir, arrays, metadata, c)
    print(f"Saved and reloaded all event arrays: {destination}", flush=True)
    print(json.dumps({"aggregate": aggregate, "validation": validation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
