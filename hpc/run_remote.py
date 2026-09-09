"""Remote preflight and four-slot manifest queue, scoped to one frozen run root."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snn_full_common import atomic_json, digest_file

def call(argv, log, backend=None):
    env = dict(os.environ)
    if backend:
        env['JAX_PLATFORMS'] = 'cuda' if backend == 'gpu' else 'cpu'
    start = time.perf_counter()
    with open(log, 'w') as output:
        result = subprocess.run([sys.executable, '-u', *map(str, argv)], env=env,
                                stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        print(Path(log).read_text()[-8000:], flush=True)
        result.check_returncode()
    return time.perf_counter() - start

def preflight(root, split, requested_backend):
    from compbio2026.data import fetch
    data = Path(fetch(split, str(root / 'data')))
    print('Dataset:', data, digest_file(data), flush=True)
    expected = {'train': '2bddb4bd46732f09982b7d1631b7c29c19853c73d3d240e3eb32bba909bdd6c1', 'test': ''}[split]
    if expected and digest_file(data) != expected:
        raise ValueError('Downloaded TRAIN differs from locally validated data')
    logs = root / 'logs'
    logs.mkdir()
    (root / 'claims').mkdir()
    (root / 'done').mkdir()
    (root / 'failed').mkdir()
    plan = []
    results = {}
    backends = ['cpu', 'gpu'] if requested_backend == 'gpu' else ['cpu']
    for model in ['adex', 'hh']:
        for backend in backends:
            output = root / f'preflight_{model}_{backend}'
            seconds = call([f'simulate_ff_rec_{model}_full.py', '--data-path', data, '--split', split,
                            '--smoke-test', '--run-dir', output], logs / f'{model}_{backend}.log', backend)
            metadata = json.loads((output / 'metadata.json').read_text())
            if metadata['status'] != 'complete':
                raise ValueError(f'Invalid preflight {output}')
            results[f'{model}_{backend}'] = {'wall_seconds': seconds, 'devices': metadata['jax_devices']}
            print(model, backend, seconds, flush=True)
        import numpy as np
        if requested_backend == 'gpu':
            with np.load(root / f'preflight_{model}_cpu/events.npz') as cpu, np.load(root / f'preflight_{model}_gpu/events.npz') as gpu:
                for key in cpu.files:
                    if key == 'metadata_json':
                        continue
                    # GPU kernels can differ from CPU by a few final floating-point
                    # bits. Keep masks/IDs/labels exact while allowing bounded
                    # round-off in numeric event arrays.
                    if cpu[key].dtype.kind == 'f':
                        atol = 1e-6 if model == 'adex' else 1e-4
                        np.testing.assert_allclose(cpu[key], gpu[key], rtol=0.0, atol=atol,
                                                   err_msg=f'CPU/GPU {model}: {key}')
                    else:
                        np.testing.assert_array_equal(cpu[key], gpu[key], err_msg=f'CPU/GPU {model}: {key}')
        call([f'param_sweep_{model}.py', '--data-path', data, '--split', split,
              '--output-dir', root / 'sweeps', '--list-configs', '--dry-run'], logs / f'{model}_manifest.log')
        manifest = next((root / 'sweeps').glob(f'*_{"AdEx" if model == "adex" else "HH"}_parameter_sweep_*'))
        plan.append({'model': model, 'manifest': str(manifest)})
    atomic_json(root / 'manifest_index.json', plan)
    atomic_json(root / 'PREFLIGHT_DONE.json', results)
    print(json.dumps(results, indent=2), flush=True)

def worker(root, slot, backend):
    """All workers atomically claim FF jobs first, then ready REC jobs.

    FF references must complete successfully. Failed claims are retained for
    explicit investigation; no silent retry or duplicate simulation occurs.
    """
    plan = json.loads((root / 'manifest_index.json').read_text())
    jobs = [(p, 'baseline_FF') for p in plan]
    jobs += [(p, c['config_id']) for p in plan for c in json.loads((Path(p['manifest'])/'sweep_manifest.json').read_text())['configurations']]
    deadline = time.monotonic() + 7 * 24 * 3600
    while time.monotonic() < deadline:
        pending = False
        worked = False
        for p, config in jobs:
            key = f"{p['model']}_{config}"
            if (root / 'done' / key).exists():
                continue
            if (root / 'failed' / key).exists():
                raise RuntimeError(f'Failed job requires investigation: {key}')
            pending = True
            if config != 'baseline_FF' and not (root / 'done' / f"{p['model']}_baseline_FF").exists():
                continue
            try:
                (root / 'claims' / key).mkdir()
            except FileExistsError:
                continue
            atomic_json(root / 'claims' / key / 'owner.json', {'worker': slot, 'pid': os.getpid(), 'backend': backend})
            selector = ['--ff-only'] if config == 'baseline_FF' else ['--config-id', config]
            print('Starting', key, flush=True)
            try:
                seconds = call([f"param_sweep_{p['model']}.py", '--manifest-dir', p['manifest'], *selector, '--execute'],
                               root / 'logs' / f'{key}.log', backend)
                output = Path(p['manifest']) / config
                if json.loads((output / 'metadata.json').read_text())['status'] != 'complete':
                    raise RuntimeError('Run is not complete')
                atomic_json(root / 'done' / key, {'seconds': seconds, 'worker': slot})
                print('Completed', key, seconds, flush=True)
            except BaseException as error:
                atomic_json(root / 'failed' / key, {'error': repr(error), 'worker': slot})
                raise
            worked = True
            break
        if not pending:
            print('All 24 jobs complete', flush=True)
            return
        if not worked:
            time.sleep(30)
    raise TimeoutError('Queue exceeded seven days; investigate retained claims')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight', 'worker'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--split', choices=['train', 'test'], required=True)
    args = parser.parse_args()
    if args.root.parent != Path('/project_ghent/rsegawa/compbio-2026/runs'):
        raise ValueError('Refusing a path outside the isolated compbio run namespace')
    if args.mode == 'preflight':
        preflight(args.root, args.split, args.backend)
    else:
        worker(args.root, args.worker, args.backend)
