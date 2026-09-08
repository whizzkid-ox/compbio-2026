"""Submit isolated compbio jobs using an existing GPULab allocation/certificate.

No URENIMOD code, environments or outputs are accessed by the containers.
The allocation name is an accounting choice, supplied explicitly by the user.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

BOOT = r'''set -euo pipefail
apt-get update -qq
apt-get install -y -qq git ca-certificates python3-venv
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
case "$RUN_ROOT" in /project_ghent/rsegawa/compbio-2026/runs/*) ;; *) exit 90;; esac
if [ "$MODE" = preflight ]; then
  mkdir -p "$(dirname "$RUN_ROOT")"
  mkdir "$RUN_ROOT"
  git clone https://github.com/whizzkid-ox/compbio-2026.git "$RUN_ROOT/code"
  git -C "$RUN_ROOT/code" checkout --detach "$COMMIT"
  python3 -m venv "$RUN_ROOT/venv"
  "$RUN_ROOT/venv/bin/pip" install 'jax[cuda12]==0.11.1' jaxley==0.14.0 numpy==2.5.2 tables==3.11.1 matplotlib==3.11.1 pandas==3.0.5 scipy==1.18.1 h5py
  "$RUN_ROOT/venv/bin/pip" freeze > "$RUN_ROOT/environment.txt"
else
  test -f "$RUN_ROOT/PREFLIGHT_DONE.json"
fi
cd "$RUN_ROOT/code"
test "$(git rev-parse HEAD)" = "$COMMIT"
export PYTHONPATH="$RUN_ROOT/code/src:$RUN_ROOT/code"
exec "$RUN_ROOT/venv/bin/python" -u hpc/run_remote.py "$MODE" --root "$RUN_ROOT" --worker "$WORKER" --backend "$BACKEND" --split "$SPLIT"
'''

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preflight', 'worker'])
    p.add_argument('--cli', required=True)
    p.add_argument('--cert', required=True)
    p.add_argument('--project', required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--worker', type=int, default=0)
    p.add_argument('--split', choices=['train', 'test'], required=True)
    p.add_argument('--backend', choices=['gpu', 'cpu'], default='gpu')
    p.add_argument('--cluster', type=int, default=11)
    p.add_argument('--submit', action='store_true')
    a = p.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', a.commit):
        p.error('Use a full immutable commit hash')
    if not re.fullmatch(r'/project_ghent/rsegawa/compbio-2026/runs/[A-Za-z0-9_-]+', a.root):
        p.error('Run root must be an isolated compbio-2026 run directory')
    if a.worker not in range(4):
        p.error('At most four worker slots are supported')
    job = {'name': f'compbio-shd-{a.split}-{a.mode}-{a.worker}',
           'description': f'Comp-bio Summer School SHD {a.split}; separate from URENIMOD research',
           'request': {'resources': {'cpus': 4, 'gpus': int(a.backend == 'gpu'), 'cpuMemoryGb': 24,
                                     **({'clusterId': a.cluster} if a.backend == 'gpu' else {})},
                       'docker': {'image': 'python:3.12-slim', 'command': ['bash', '-c', BOOT],
                                  'storage': [{'hostPath': '/project_ghent', 'containerPath': '/project_ghent'}],
                                  'environment': {'MODE': a.mode, 'RUN_ROOT': a.root, 'COMMIT': a.commit,
                                                  'WORKER': str(a.worker), 'BACKEND': a.backend,
                                                  'SPLIT': a.split}},
                       'scheduling': {'interactive': False, 'restartable': False,
                                      'minDuration': '10 minutes',
                                      'maxDuration': '6 hours' if a.mode == 'preflight' else '14 days'}}}
    destination = Path(__file__).parent / 'submissions' / Path(a.root).name
    destination.mkdir(parents=True, exist_ok=True)
    spec = destination / f'{a.mode}_{a.worker}.json'
    # Refuse accidental duplicate submissions of this exact slot.
    receipt = spec.with_suffix('.receipt.txt')
    if a.submit and receipt.exists() and re.search(r'[0-9a-f]{8}-[0-9a-f-]{27,}', receipt.read_text(encoding='utf-8')):
        raise FileExistsError(f'An accepted job receipt already exists: {receipt}')
    spec.write_text(json.dumps(job, indent=2), encoding='utf-8')
    print(spec)
    if a.submit:
        result = subprocess.run([a.cli, '--cert', a.cert, 'submit', '--project', a.project, str(spec)],
                                capture_output=True, text=True)
        receipt.write_text(result.stdout + result.stderr, encoding='utf-8')
        print(result.stdout, result.stderr)
        result.check_returncode()

if __name__ == '__main__':
    main()
