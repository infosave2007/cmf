#!/usr/bin/env python3
"""Measure the native CMF Qwen Image CLI; Python never runs model inference."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import threading
import shutil

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--cortiq', type=Path, required=True)
p.add_argument('--model', type=Path, required=True, help='Ready CMF file or directory with the three CMF components')
p.add_argument('--image', type=Path, required=True)
p.add_argument('--prompt', required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--steps', type=int, default=30)
p.add_argument('--width', type=int, default=512)
p.add_argument('--height', type=int, default=512)
p.add_argument('--cfg', type=float, default=4.0)
p.add_argument('--seed', type=int, default=7)
p.add_argument('--reference-size', type=int, default=1024)
p.add_argument('--backend', choices=['metal', 'wgpu', '0'], default='metal')
p.add_argument('--lock-file', type=Path)
p.add_argument('--profile', action='store_true')
a = p.parse_args()
a.out.parent.mkdir(parents=True, exist_ok=True)
log = a.out.with_suffix('.log')
receipt_path = a.out.with_suffix('.json')
cmd = [str(a.cortiq.resolve()), 'imagine', str(a.model.resolve()), '--image', str(a.image.resolve()),
       '--prompt', a.prompt, '--steps', str(a.steps), '--width', str(a.width), '--height', str(a.height),
       '--cfg', str(a.cfg), '--seed', str(a.seed), '--reference-size', str(a.reference_size), '--out', str(a.out.resolve())]
env = os.environ.copy()
env['CMF_GPU'] = a.backend
# Fix the execution arm so repeated seeds do not alternate CPU/GPU probe samples.
env['CMF_GPU_PROBE'] = '0'
if a.backend == 'wgpu':
    env.setdefault('XDG_RUNTIME_DIR', '/tmp')
    env['WGPU_BACKEND'] = 'vulkan'
if a.profile:
    env['CMF_QWEN_IMAGE_PROFILE'] = '1'
version = subprocess.check_output([str(a.cortiq.resolve()), '--version'], text=True).strip()
binary_sha = hashlib.sha256(a.cortiq.read_bytes()).hexdigest()
gpu_samples = []
stop_sampling = threading.Event()
gpu_info = None
if sys.platform == 'linux' and a.backend == 'wgpu' and shutil.which('nvidia-smi'):
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                        '--format=csv,noheader,nounits'], text=True).strip()
def sample_gpu():
    while not stop_sampling.is_set():
        try:
            row = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                           '--format=csv,noheader,nounits'], text=True, timeout=5).splitlines()[0]
            memory, utilization = (float(x.strip()) for x in row.split(','))
            gpu_samples.append((memory, utilization))
        except (OSError, ValueError, subprocess.SubprocessError, IndexError):
            pass
        stop_sampling.wait(1.0)
lock = a.lock_file.open('a') if a.lock_file else contextlib.nullcontext()
with lock as lock_handle:
    if a.lock_file:
        import fcntl
        print(f'Waiting for GPU lock {a.lock_file}', flush=True)
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
    measured_cmd = (['/usr/bin/time', '-l', *cmd] if sys.platform == 'darwin' else
                    ['/usr/bin/time', '-v', *cmd] if Path('/usr/bin/time').exists() else cmd)
    sampler = threading.Thread(target=sample_gpu, daemon=True) if gpu_info else None
    if sampler:
        sampler.start()
    started = time.time()
    print('Native generation started; log: ' + str(log), flush=True)
    with log.open('w') as out:
        try:
            result = subprocess.run(measured_cmd, env=env, stdout=out, stderr=subprocess.STDOUT)
        finally:
            stop_sampling.set()
            if sampler:
                sampler.join(timeout=6)
    elapsed = time.time() - started
text = log.read_text(errors='replace')
stages = []
for line in text.splitlines():
    match = re.match(r'^(text encoder|reference VAE|denoiser|decode VAE): (\d+)/(\d+) \(([\d.]+)s\)$', line)
    if match:
        stages.append(dict(stage=match[1], completed=int(match[2]), total=int(match[3]), elapsed_s=float(match[4])))
peak = re.search(r'^\s*(\d+)\s+maximum resident set size\s*$', text, re.M)
linux_peak = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', text)
receipt = dict(command=cmd, version=version, binary_sha256=binary_sha, exit_code=result.returncode,
               elapsed_s=elapsed, backend=a.backend, gpu_probe='0', stages=stages,
               engine_env={k: env[k] for k in (
                   'CMF_THREADS', 'CMF_QWEN_IMAGE_PROFILE', 'CMF_QWEN_IMAGE_FUSED_QKV',
                   'CMF_QWEN_IMAGE_FUSED_MLP', 'CMF_QWEN_IMAGE_FUSED_MLP_COOP', 'CMF_QWEN_VAE_GPU', 'CMF_GPU_UPLOAD',
                   'CMF_GPU_UPLOAD_CHUNK_MB', 'CMF_PLANE_CACHE_MB', 'CMF_COOP',
                   'CMF_GPU_VRAM_MB', 'CMF_RAM_TIER_MB', 'WGPU_BACKEND') if k in env},
               max_rss_bytes=int(peak[1]) if peak else int(linux_peak[1]) * 1024 if linux_peak else None,
               gpu_info=gpu_info, gpu_peak_used_mib=max((x[0] for x in gpu_samples), default=None),
               gpu_utilization_mean=sum(x[1] for x in gpu_samples) / len(gpu_samples) if gpu_samples else None, output=str(a.out.resolve()),
               settings=dict(width=a.width, height=a.height, steps=a.steps, cfg=a.cfg, seed=a.seed, reference_size=a.reference_size))
if result.returncode == 0 and a.out.is_file():
    receipt['output_sha256'] = hashlib.sha256(a.out.read_bytes()).hexdigest()
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
sys.exit(result.returncode)
