"""Run everything still outstanding, one experiment at a time.

Waits for any other venv python (notably the extended geofence run) to
finish first, so the machine is never oversubscribed. exp1 comes first
because it was interrupted; the sensitivity sweep goes last because it
is by far the longest and partial progress before it is still useful.
"""
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / '.venv-cf' / 'Scripts' / 'python.exe')
LOG = ROOT / 'rerun_log.txt'
SELF = 'run_remaining'

ORDER = ['exp2', 'exp1', 'baselines', 'ablation', 'learning', 'noise', 'sensitivity']


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def others_running():
    out = subprocess.run(
        ['powershell', '-NoProfile', '-Command',
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
         "| ForEach-Object { $_.CommandLine }"],
        capture_output=True, text=True).stdout or ''
    return [l for l in out.splitlines() if '.venv-cf' in l and SELF not in l]


log(f'{SELF}: waiting for the machine to clear')
while others_running():
    time.sleep(60)

failures = []
for step in ORDER:
    log(f'=== START {step} ===')
    t0 = time.time()
    p = subprocess.run([PY, '-u', 'overnight_step.py', step], cwd=str(ROOT),
                       capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    tail = (p.stdout + p.stderr)[-300:].replace('\n', ' | ')
    ok = p.returncode == 0
    log(f'=== {"OK" if ok else "FAIL"} {step} rc={p.returncode} in {dt:.1f} min :: {tail}')
    if not ok:
        failures.append(step)
        # A step that dies instantly with no output was killed rather than
        # having failed on its own merits (session teardown does this).
        # Stop instead of burning through the rest of the queue the same way.
        if dt < 0.2 and not tail.strip():
            log('step died instantly with no output; aborting queue')
            break

log(f'ALL REMAINING COMPLETE. failures: {failures or "none"}')
