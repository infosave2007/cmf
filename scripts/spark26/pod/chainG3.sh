#!/bin/bash
# chainG3.sh: after chainG2, the tiled graph's GPU stage profile (`CMF_GPU_TS=1 CMF_BATCH_TS=1`)
# at 4096 and 16000 tokens with coop GEMMs and 512-row submits, and a 32-token decode after
# each path to see whether decode after the graph prefill matches decode after the walk.
# Output logs/g3_*.log, logs/g3_decode.txt, chainG3.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
until [ -f logs/chainG2.done ]; do sleep 5; done
F=hf/cmf/spark-x2.6-4b-q4tp.cmf; B=./cortiq.xt
bash quiet.sh bash -c "
for n in 4096 16000; do
  CMF_DENSE_GRAPH_PREFILL=1 CMF_BATCH_COOP=1 CMF_BATCH_K=512 CMF_GPU_TS=1 CMF_BATCH_TS=1 CMF_BATCH_KERNEL_TS=1 CMF_GRAPH_PROF=1 $B bench $F --core --ignore-eos --tokens 4 --ctx \$n > logs/g3_prof_\$n.log 2>&1
done
: > logs/g3_decode.txt
for v in walk:CMF_DENSE_GRAPH_PREFILL=0 graph:CMF_DENSE_GRAPH_PREFILL=1,CMF_BATCH_COOP=1,CMF_BATCH_K=512; do
  name=\${v%%:*}; envs=\${v#*:}
  env \${envs//,/ } $B bench $F --core --ignore-eos --tokens 32 --ctx 4096 > logs/g3_dec_\$name.log 2>&1
  echo \"4096 | \$name | \$(grep -E 'Prompt|Decode' logs/g3_dec_\$name.log | tr '\n' ' ' | tr -s ' ' | cut -c1-240)\" >> logs/g3_decode.txt
done
"
touch logs/chainG3.done
