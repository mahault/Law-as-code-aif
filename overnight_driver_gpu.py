"""GPU pipeline driver (WSL2/Ubuntu, RTX A4000): sequential, subprocess-isolated.

Same structure as overnight_driver.py but runs inside WSL with the CUDA venv.
The committed-model T=10 results are already snapshotted in results_t10_committed/.
"""
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = '/root/lawenv/bin/python'
LOG = ROOT / 'overnight_log_gpu.txt'

ORDER = ['exp3', 'baselines', 'ablation', 'exp2', 'exp1', 'learning', 'noise', 'sensitivity']

ENV = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false')


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


log('GPU driver started')
failures = []
for step in ORDER:
    for attempt in (1, 2):
        log(f'START {step} (attempt {attempt})')
        t0 = time.time()
        p = subprocess.run([PY, '-u', 'overnight_step.py', step], cwd=str(ROOT),
                           capture_output=True, text=True, env=ENV)
        dt = (time.time() - t0) / 60
        tail = (p.stdout + p.stderr)[-400:].replace('\n', ' | ')
        log(f'END {step} attempt {attempt}: rc={p.returncode} in {dt:.1f} min :: {tail}')
        if p.returncode == 0:
            break
        time.sleep(120)
    else:
        failures.append(step)

log(f'PIPELINE COMPLETE. failures: {failures or "none"}')
