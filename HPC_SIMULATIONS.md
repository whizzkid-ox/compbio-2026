# Full SHD simulations and HPC sweeps

The full runners select every English label 0–9 in ascending source-trial order from the requested train/test HDF5. The local train file contains 8,156 total trials, of which **4,011 are English**. No full dataset or parameter-sweep simulation was run locally during this implementation. The test HDF5 was not available locally.

## Files

| File | Purpose |
|---|---|
| `simulate_ff_rec_adex.py` | Renamed AdEx diagnostic prototype; ordinary, seven-point sweep and 50-trial post-hoc scientific code preserved. |
| `simulate_ff_rec_hh.py` | Unchanged HH diagnostic prototype, including its existing 3e-5 µS ordinary default. |
| `simulate_ff_rec_adex_full.py` | All-English AdEx entry point, default input_g 3e-5 µS. |
| `simulate_ff_rec_hh_full.py` | All-English HH entry point, default input_g 3e-4 µS. |
| `param_sweep_adex.py`, `param_sweep_hh.py` | Generate immutable OFAT manifests; explicitly execute one selected HPC job. |
| `snn_full_common.py` | Shared selection, checkpoint/export, validation and manifest orchestration; imports the corresponding prototype's numerical model. |
| `test_snn_full.py` | Selection, isolation, storage, failure, locking and compatibility tests. |

Deploy the scripts together with their existing course/data dependencies. Use a site-supported JAX/Jaxley environment; no package upgrades or GPU installation were performed here. Execution records package versions, source hashes, backend and devices. The locally tested environment used Jaxley 0.14.0 and JAX/JAXlib 0.11.1.

## Fixed baseline

| Parameter | Full-run default |
|---|---|
| Conditions / selection | FF and REC / all English trials in requested split |
| Input conductance | AdEx 3e-5; HH 3e-4 µS per synapse |
| Population | 700 inputs; 80 excitatory + 20 inhibitory outputs |
| Input wiring | Tonotopic; fan_in 60; Gaussian width 35 channels |
| Recurrent wiring | Probability 0.10; no self-connections; fixed seed realization |
| Recurrent conductances | E 3e-5; I 1.2e-4 µS; I/E ratio 4 |
| Synaptic decay | Input, recurrent E and recurrent I: 5 ms |
| Reversal potentials | Excitatory 0; inhibitory −80 mV |
| Synaptic activation | Input and recurrence: threshold −35 mV, slope 2 mV |
| Geometry / capacitance | Radius 10 µm, length 10 µm; 1 µF/cm² |
| Initial voltage | −70 mV for every output, SD 0 |
| Timing | dt 0.1 ms; settling 50 ms; input 800 ms; post-input 100 ms |
| Replay waveform | Rest −70 mV; +20 mV pulses of 1 ms |
| Master seed | 2026; separate selection/input/recurrent/initial seeds derived by the existing prototype and stored explicitly |
| Diagnostics / checkpoints | 0 detailed diagnostic trials; commit every 25 attempted trials |

AdEx retains the installed-model defaults E_L −70 mV, v_T −50 mV, delta_T 2 mV, spike threshold 0 mV and reset −58 mV, with g_L 5e-5 S/cm², a 1e-5 S/cm², b 0, tau_w 100 ms and initial adaptation 0. The existing unit-conversion adapter remains unchanged. Its model spike flag is authoritative; −20 mV crossings are diagnostic.

HH retains gNa 0.12, gK 0.036 and gLeak 0.0003 S/cm², with reversals +50, −77 and −54.3 mV. Gates start at their voltage-dependent equilibrium at −70 mV. HH events use the existing upward −20 mV crossing detector. There is no HH phenotype fitting, input recalibration, imposed output waveform, added delay or added refractory period.

Complete scientific configurations, derived seeds and intrinsic parameters are in each manifest/run metadata. Full-run CLI overrides are available through `--help`; stored scientific settings cannot be changed on resume.

## Eleven default configurations

All rows retain fan_in 60, width 35 and model-specific input_g. Conductances below are µS. These are OFAT settings, not a factorial grid.

| Index | config_id | p_rec | E conductance | I conductance | I/E |
|---:|---|---:|---:|---:|---:|
| 0 | baseline | .10 | 3e-5 | 1.2e-4 | 4 |
| 1 | rec_probability_0p00 | 0 | 3e-5 | 1.2e-4 | 4 |
| 2 | rec_probability_0p05 | .05 | 3e-5 | 1.2e-4 | 4 |
| 3 | rec_probability_0p20 | .20 | 3e-5 | 1.2e-4 | 4 |
| 4 | rec_probability_0p30 | .30 | 3e-5 | 1.2e-4 | 4 |
| 5 | rec_gain_0p5 | .10 | 1.5e-5 | 6e-5 | 4 |
| 6 | rec_gain_2 | .10 | 6e-5 | 2.4e-4 | 4 |
| 7 | rec_gain_4 | .10 | 1.2e-4 | 4.8e-4 | 4 |
| 8 | EI_ratio_1 | .10 | 3e-5 | 3e-5 | 1 |
| 9 | EI_ratio_2 | .10 | 3e-5 | 6e-5 | 2 |
| 10 | EI_ratio_8 | .10 | 3e-5 | 2.4e-4 | 8 |

The baseline represents p=.10, gain=1 and I/E=4 only once. Ratio changes also change total recurrent conductance; they are not total-conductance-normalized manipulations.

The default plan has one separate `baseline_FF` job and these 11 REC jobs: **12 jobs per model per split**. Each REC archive points to the same verified FF reference. `--rerun-ff` generates a plan with both conditions in each row and no standalone FF job.

Implemented but disabled grids: fan_in [30,60,90,120] and width [15,35,70,140]. New manifests can explicitly enable them with `--enable-fan-in` and/or `--enable-width`. Each adds three non-baseline rows; enabling both gives 17 unique configurations. Each input-wiring deviation runs both FF and REC. No input-wiring sweep simulation was executed locally.

## Commands

Examples below are for an activated HPC Python environment, from the script directory. Supply the actual dataset and scratch paths. `--data-path` accepts the HDF5 location supported by the prototype (for example the directory containing `hdspikes/shd_train.h5`).

Read-only planning and tiny hardware validation:

```bash
python simulate_ff_rec_adex_full.py --data-path /scratch/shd/data --split train --plan-only
python simulate_ff_rec_hh_full.py --data-path /scratch/shd/data --split test --plan-only
python simulate_ff_rec_adex_full.py --data-path /scratch/shd/data --smoke-test --diagnostic-trials 0 --run-dir /scratch/validation/adex
python simulate_ff_rec_hh_full.py --data-path /scratch/shd/data --smoke-test --diagnostic-trials 0 --run-dir /scratch/validation/hh
python simulate_ff_rec_hh_full.py --compare-runs /scratch/validation/adex /scratch/validation/hh
```

The smoke runs use two deterministic source trials; `--trial-ids ID1,ID2` optionally selects one or two explicit English trials and requires `--smoke-test`. The cross-model comparison checks shared trial identities, labels, speakers, replay hashes, wiring, initial voltages, geometry, timing and intentional baseline input_g differences. It does not require equal output spikes.

Full baseline runs, **HPC only**:

```bash
python simulate_ff_rec_adex_full.py --data-path /scratch/shd/data --split train --diagnostic-trials 0 --checkpoint-trials 25 --run-dir /scratch/results/adex_train
python simulate_ff_rec_hh_full.py --data-path /scratch/shd/data --split train --diagnostic-trials 0 --checkpoint-trials 25 --run-dir /scratch/results/hh_train
# Repeat with --split test and distinct output directories for test data.
# Resume exactly the stored experiment:
python simulate_ff_rec_adex_full.py --resume /scratch/results/adex_train
```

Generate plans on HPC using HPC dataset paths; generation does not simulate:

```bash
python param_sweep_adex.py --data-path /scratch/shd/data --split train --output-dir /scratch/results --list-configs --dry-run
python param_sweep_hh.py --data-path /scratch/shd/data --split train --output-dir /scratch/results --list-configs --dry-run
```

Each prints a unique manifest directory. Use a separate manifest for each model and split. Do not edit generated manifests; their hashes are checked. For one AdEx manifest:

```bash
MANIFEST=/scratch/results/REPLACE_WITH_PRINTED_AdEx_parameter_sweep_DIRECTORY
python param_sweep_adex.py --manifest-dir "$MANIFEST" --ff-only --dry-run
python param_sweep_adex.py --manifest-dir "$MANIFEST" --task-index 2 --dry-run
# HPC execution: FF must finish successfully before REC jobs begin.
python param_sweep_adex.py --manifest-dir "$MANIFEST" --ff-only --execute
python param_sweep_adex.py --manifest-dir "$MANIFEST" --config-id rec_probability_0p05 --execute
# Restart an interrupted selected job:
python param_sweep_adex.py --manifest-dir "$MANIFEST" --task-index 2 --execute --resume
```

Execution requires `--execute` plus a selector; there is no implicit run-all mode. `--dry-run` suppresses execution even if `--execute` is present. Generation parameters are rejected with `--manifest-dir`; use the saved plan unchanged.

## Example SLURM structure (not submitted)

Use one GPU per configuration as an initial strategy, benchmark first, and set partition, memory and walltime according to the site and measured resource use. The Python environment must already be available within each job.

Save this example as `rec_array.sbatch` on HPC:

```bash
#!/bin/bash
set -euo pipefail
DRIVER="$1"
MANIFEST="$2"
python "$DRIVER" --manifest-dir "$MANIFEST" --task-index "$SLURM_ARRAY_TASK_ID" --execute
```

Submit the baseline FF job and make the REC array depend on its success:

```bash
DRIVER=param_sweep_adex.py  # use param_sweep_hh.py with its HH manifest
FF_JOB=$(sbatch --parsable --gres=gpu:1 --job-name=shd_ff \
  --wrap="python '$DRIVER' --manifest-dir '$MANIFEST' --ff-only --execute")
sbatch --gres=gpu:1 --array=0-10 --dependency="afterok:${FF_JOB%%;*}" \
  --job-name=shd_rec rec_array.sbatch "$DRIVER" "$MANIFEST"
```

These are instructions only; no jobs were submitted. The standalone FF job is outside array indices 0–10. Retry selected interrupted tasks explicitly with `--resume`. Optional input grids change the array bounds; derive them from `--list-configs`. No trial sharding or multi-trial batching is implemented.

## Output and recovery

```text
<timestamp>_<model>_parameter_sweep_<uuid>/
  sweep_manifest.json          # all scientific parameters, seeds, source hashes
  sweep_manifest.csv           # one row per unique configuration
  sweep_index.json             # archive, FF reference and status-file mapping
  job_status/<config_id>.json  # separate per-job status; no shared write race
  baseline_FF/
    metadata.json              # configuration, identities, journal, validation
    shared.npz                 # full planned trial roster and network arrays
    checkpoints/chunk_<index>_<uuid>.npz
    events.npz                 # aggregated committed event archive
    trials.csv                 # compact per-trial, per-condition metrics
    run.lock                   # persistent file; OS lock released on process exit
    COMPLETE.txt               # or PARTIAL.txt
    ERROR_<uuid>.json          # present on unexpected failure
  baseline/                    # REC result, with same run-file structure
  rec_probability_0p00/
  ...
```

Full runs outside a sweep use the same run-file structure. Detailed traces are saved only for an explicitly requested diagnostic subset. Full-dataset voltage traces are never accumulated or stored. Checkpoints retain compressed events; final export streams through disk-backed arrays into a normal, pickle-free NPZ. Budget space for checkpoints, final output and temporary uncompressed event arrays during export.

Each condition C has `C_times_ms`, `C_neuron_ids`, `C_offsets`, analogous `C_pre_*` settling arrays, and `C_valid`. Input-relative event times include the post-input period. IDs index the saved neuron roster/population. The archive also includes trial IDs, labels, speakers, source split, order, model, conditions, connectivity, receptive-field centres, initial voltages, stimulus hashes and metadata JSON.

```python
import numpy as np
with np.load("events.npz", allow_pickle=False) as z:
    k, condition = 0, "REC"
    assert z[condition + "_valid"][k], "Numerically failed trial: missing data"
    start, stop = z[condition + "_offsets"][k:k+2]
    times = z[condition + "_times_ms"][start:stop]
    neurons = z[condition + "_neuron_ids"][start:stop]
    trial_id, label = z["trial_ids"][k], z["labels"][k]
```

An empty slice is valid silence only when its validity mask is true. A partial archive contains the committed prefix; `planned_trial_ids` retains the complete requested roster. Recognized numerical failures have false masks and null metrics, never fabricated zero firing. They remain recorded as attempted trials and are not automatically retried on resume. A completed pass containing such failures stays PARTIAL and exits with code 2, blocking an `afterok` FF dependency. Unexpected programming/API/assertion errors write an error report and fail loudly.

Immutable chunks are atomically written and verified before the journal advances. Resume verifies committed chunk checksums and restores the stored selection/configuration. Uncommitted orphan chunks are ignored. A crash may require repeating up to one uncommitted checkpoint block. `--stop-after N` offers a deliberate partial stop after N new trials; it exits successfully if there were no numerical failures, but the result is still PARTIAL and must be resumed. Normal creation refuses existing run directories.

Resume requires matching scientific settings, source/data hashes, package versions and backend identity. Keep the deployed code/environment available; changed code or backend requires a new run. A completed resume verifies the archive and exits without simulation. Filesystems must support advisory locking and atomic same-directory replacement; check the HPC filesystem's guarantees. Metadata/journal is authoritative if a process is killed between output/marker updates.

## GPU assessment and scientific limits

The scripts do not force CPU or a CUDA device. Installed Jaxley selects its available CPU/GPU solver path and JAX executes the existing compiled integrator on the available backend. Conductance, gate, synapse and solver arithmetic can benefit from GPU execution. **Substantial speedup is not established**, particularly for only 100 output neurons. JAX performance depends on workload size, compilation and transfer costs ([JAX performance FAQ](https://docs.jax.dev/en/latest/faq.html#is-jax-faster-than-numpy)).

Likely bottlenecks from local source inspection are sequential time integration (9,500 steps), first compilation, HDF5/NumPy replay preparation, compression/checksums and host/device transfers. The prototype materializes each trial's recordings on the host: roughly 23 MB for AdEx or 30 MB for HH per condition, in addition to about 53 MB of replay waveform data. Float64 throughput varies widely across GPUs. Host conversion synchronizes JAX, so benchmark warmed complete trials, not asynchronous dispatch alone ([JAX asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)).

One GPU per configuration is a reasonable initial scheduling unit, not a measured optimum. Benchmark the two-trial smoke path on the actual GPU before committing resources. Multi-trial batching could amortize overhead but increases memory and has not been validated; retain the current unbatched path. No batching speedup or cross-device bitwise reproducibility is claimed.

CPU smoke checks use exact repeat/reset event equality. Separate zero-recurrence FF/REC compiled graphs can differ near an HH spike: the full HH control allows 1e-4 mV voltage error and 1e-6 gate error while still requiring identical crossing masks. Observed smoke differences were 1.4634e-6 mV and 1.332e-8 respectively. An initial stricter comparator aborted before any trial commit; its error/partial record was retained. Prototype tolerances and numerical models were unchanged.

The baselines are provisional operating points, not validated biological fits. Matching connectivity does not imply matching recurrent drive: the models use different evolving presynaptic voltage waveforms. No decoder accuracy, phenotype equivalence, timestep convergence or GPU equivalence was established in this change.

## Local verification actually executed

- Syntax compilation and `--help` for all six entry points; helper/test compilation.
- Read-only train selection plans for both models: all 4,011 English IDs, no subsampling.
- Both default manifest generations, CSV/JSON checks, exactly 11 unique OFAT configurations, selector/FF dry runs and input-parameter isolation.
- Two-trial ordinary AdEx and HH prototype regressions: all non-metadata archive arrays exactly matched prior validated runs. AdEx source AST unchanged after filename normalization; HH source byte-identical. Existing seven-point/50-trial modes were preserved, not rerun this turn.
- Two-trial AdEx full run interrupted deliberately after one committed trial, then resumed: final events exactly matched the prototype.
- Two-trial HH full run at 3e-4 µS: events exactly matched the prior validated 3e-4 diagnostic-sweep result.
- Cross-model shared-input/initialization/geometry/timing validation passed. Archive schemas, offsets and no-object-array loading passed.
- Eight unit tests passed, including synthetic injected numerical/programming failures, invalid-versus-silent masks, checkpoint corruption/orphan handling, exclusive locking, resume guards and optional-grid generation. Optional-grid tests generated configurations/connectivity only.
- Existing-directory and changed-parameter resume refusals checked. No full English train/test simulation, 11-configuration simulation or fan-in/width simulation was run locally. Two GPU preflight attempts were cancelled during CUDA dependency installation; the successful submitted runs use CPU-only JAX.

Logs and tiny-run archives are under `results/full_implementation_validation/`; the final unit-test log is `unit_tests_final.log`.

## GPULab deployment (2026-09-09)

The actual available service is GPULab, not SLURM. `hpc/gpulab.py` creates or submits an explicit preflight/worker container using the existing certificate and accounting allocation. It does not modify that allocation. All scientific code, Python environment, data and outputs are isolated beneath `/project_ghent/rsegawa/compbio-2026/runs/<run-id>`; no URENIMOD code or output directories are used. Every job checks out an immutable Git commit. Credentials are never embedded in job files; this public GitHub fork requires no clone token.

Preflight installs pinned JAX/Jaxley and dependencies in a new run-local environment, downloads the requested split and checks the TRAIN SHA256 against the locally validated dataset. GPU preflight mode also compares two-trial CPU/GPU runs with bounded floating-point tolerances; CPU-only mode runs the same smoke checks on `cpu:0` and skips CUDA packages. Five workers per split share an atomic claim queue: the two model-specific baseline FF runs execute once, followed by their eleven REC configurations. REC jobs wait for successful FF completion. Each worker requests 10 CPU cores; ten worker jobs are submitted across TRAIN and TEST, so GPULab queues any jobs beyond the currently available capacity. No sharding or concurrent trial batching is introduced.

Use `python hpc/gpulab.py --help` for explicit certificate, allocation, commit, root and submission options. The default action only saves a job specification. Submission receipts remain local under ignored `hpc/submissions/`. Remote logs, claims, done/failed markers, manifests and run metadata allow auditing. A failed or interrupted claim is retained and requires explicit investigation; the launcher does not automatically resubmit or clear it. TRAIN is the initial sensitivity-analysis split; TEST is reserved for subsequent held-out decoder evaluation.

The active CPU run roots are `/project_ghent/rsegawa/compbio-2026/runs/shd_train_20260909_cpu2_0b7cf67` and `/project_ghent/rsegawa/compbio-2026/runs/shd_test_20260909_cpu2_0b7cf67`, both checked out at commit `0b7cf67fe9cf5725abe7e8c137d7c0d3fa145ec6`. They use the `urenimod` accounting allocation only; all code, environments, data and outputs remain in the compbio-2026 namespace. Repository deployment also supports both repository-root and parent-directory layouts for locating course utilities/data. The old `simulate_ff_rec.py` filename is a small compatibility launcher for the renamed AdEx prototype.
