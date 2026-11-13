#!/bin/bash
# download Edge0 sources: 145B-NVFP4 whole; 120B-NVFP4 only its own files (the rest are byte-identical)
cd /root/e0/src
H=/root/venv/bin/hf
$H download Edge0/Edge0-145B-NVFP4 --local-dir e145n --exclude "assets/*" > /root/e0/logs/dl145n.log 2>&1; echo "145n rc=$?"
$H download Edge0/Edge0-120B-NVFP4 --local-dir e120n --include "experts.bin" "latent_dense_q4.bin" "latent_dense_q8.bin" "model.json" "native_experts.txt" "index.txt" "README.md" "LICENSE" "NOTICE" "release-scope.json" > /root/e0/logs/dl120n.log 2>&1; echo "120n rc=$?"
$H download Edge0/Edge0-145B-q3k --local-dir e145q --include "experts.bin" "model.json" "native_experts.txt" "index.txt" "model-bench.json" "scales/*" > /root/e0/logs/dl145q.log 2>&1; echo "145q rc=$?"
ls -la e145n e120n e145q; df -h /
echo DL-DONE
