#!/bin/bash
# vmmprobe.py under the card's lock (a context of its own, 512 MB blocks).
cd /root/h3i || exit 1
flock gpu.lock timeout 300 venv/bin/python scripts/vmmprobe.py
