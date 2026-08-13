"""Run the extended geofence experiment once the machine is free."""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / '.venv-cf' / 'Scripts' / 'python.exe')
LOG = ROOT / 'rerun_log.txt'

def log(m):
    line = f'[{time.strftime("%H:%M:%S")}] {m}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def busy():
    out = subprocess.run(['powershell','-NoProfile','-Command',
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.CommandLine }"],
        capture_output=True, text=True).stdout or ''
    return [l for l in out.splitlines()
            if '.venv-cf' in l and 'run_exp2_v2' not in l and 'rerun_rest' not in l]

while busy():
    time.sleep(60)

log('=== START exp2_v2 (geofence + emergency exception) ===')
t0 = time.time()
p = subprocess.run([PY, '-u', 'overnight_step.py', 'exp2'], cwd=str(ROOT),
                   capture_output=True, text=True)
dt = (time.time() - t0) / 60
Path(ROOT / 'exp2_v2_stdout.txt').write_text(p.stdout + p.stderr, encoding='utf-8')
log(f'=== {"OK" if p.returncode==0 else "FAIL"} exp2_v2 in {dt:.1f} min ===')
