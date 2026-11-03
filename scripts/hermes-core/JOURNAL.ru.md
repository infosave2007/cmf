# Hermes core: pod journal

Pod: `ssh root@81.27.69.177 -p 28634`, RTX PRO 6000 Blackwell Server 97.9 GB, EPYC 9554 with a 27 CPU quota, 188 GB cgroup.
Sources `/root/cmf` (origin/master d979ebfe + trainer patch), runs `/root/runs/hermes`, corpus `/root/corpus`, teacher `/root/models/qwen38-27b-q4tp.cmf` (sha256 eec6a359 as on HF).

## 03.11 trainer

step-bench, in-process timers, teacher and scenarios frozen with SIGSTOP (a teacher finishing a request doubles the step).

| What | 340M B8 T1024, ms/step |
|---|---|
| master | 1340 |
| DLG dots in their own pass (bit-identical loss) | 1211 |
| route from GEMM products (0 flips of 256, residual 7.7e-7) | 1123 |
| 128-entry only from 400 tiles, split-K to 752 tiles on GB202 | 1003 |

Rejected, measured: gemm_r8 (8×8 lane, no workgroup memory) 3189 ms, bound by the bus at 2 flops per byte; gemm128x (16×16 lane, 64 threads) 2602 ms; all 128 entry 1498 ms; all 64 entry +0 %.

Remaining reserve, unexploited: encode 75 ms/step (buffer + bind group per op), CONV_BWD 41 ms, other short ops about 60 ms.

Final speeds, tok/s: 56M B16 32.5k; 164M B16 13.9k; 245M B16 11.2k; 340M B8 8.2k, B16 9.6k; 451M B8 6.9k; 513M B8 6.4k. 513M B16 and 340M B32 run out of memory with the teacher on the card. Above 536M the arena exceeds the 2 GB wgpu binding: 881M does not start. hidden not a multiple of 64 falls to the slow tile path (544: 7.7k at 141M).

Tests on the pod: lib 12, vulkan_anchor_batches 4, correctness 5, gdn_wy 3, gemm64 2, operator_contract 11, training 4 pass; terminal_sft_final_holdout_eval needs CMF_SFT_FINAL_CKPT.

## 03.11 Hermes

Hermes d7572ce needs Python 3.14 (dependency pins are marked for it, a 3.11 venv imports nothing). Tools go through the ssh backend to unprivileged users scen0..3; /root is 700. unshare and mount are refused, chroot and proot work. Real request: 12.5k chars of system prompt plus 39k chars of 20 tool schemas, about 12.5k tokens every call, full prefill each call (35 s) because resident GDN state cannot roll back. A 256-token default cap ended replies during reasoning: --max-tokens 4096.

## 03.11 corpus

Licences in scripts/hermes-core/LICENSES.tsv. Best answer per question: unix 151k pairs 0.30 G chars, askubuntu 194k 0.35, serverfault 202k 0.39, superuser 271k 0.42. Docs: man 29 M chars (3101 pages), tldr 5, powershell 33, arch wiki 18, kubernetes 15, python 8, docker 8, hermes 8, freebsd 8, git 3, debian handbook 1.
