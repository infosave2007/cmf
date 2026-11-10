#!/bin/bash
# Planted facts in long Wikipedia text on the CUDA path
# (scripts/spark26/passkey_file.py: a pass key, an access code, a password,
# a room number at a random place, the question at the end, eight trials a
# length): the far field, and exact attention over the same prompts.
#   chain_cuda_passkey.sh <binary> [chars ...]
root=${CUDA_RUN_ROOT:-/root/cuda}
cd "$root" || exit 1
bin=${1:-bin/cortiq-cuda}
shift
[ $# -eq 0 ] && set -- 60000 120000 240000
m=models/spark-x2.6-4b-q4tp-turbo.cmf
py=./venv/bin/python
if [ ! -s data/wiki.test.raw ]; then
  # WikiText's raw test split, the text the turbo page's table was measured on
  mkdir -p data
  ./venv/bin/pip -q install pyarrow
  ./venv/bin/hf download Salesforce/wikitext wikitext-103-raw-v1/test-00000-of-00001.parquet --repo-type dataset --local-dir data/wt > /dev/null
  $py -c "import pyarrow.parquet as pq; t = pq.read_table('data/wt/wikitext-103-raw-v1/test-00000-of-00001.parquet'); open('data/wiki.test.raw', 'w', encoding='utf-8').write(''.join(t.column('text').to_pylist()))"
fi
wc -c data/wiki.test.raw
export PASSKEY_FILE=$root/out/passkey_prompt.txt
for n in "$@"; do
  echo "== far field, $n characters"
  $py src/scripts/spark26/passkey_file.py "$bin" $m data/wiki.test.raw $n 8 CMF_GPU=cuda CMF_MAX_SEQ=131072 CMF_GPU_PROBE=0 2>&1 | tail -9
  echo "== exact attention, $n characters"
  $py src/scripts/spark26/passkey_file.py "$bin" $m data/wiki.test.raw $n 8 CMF_GPU=cuda CMF_O1=off CMF_MAX_SEQ=131072 CMF_GPU_PROBE=0 2>&1 | tail -9
done
echo CHAIN_CUDA_PASSKEY_DONE
