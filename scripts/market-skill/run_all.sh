#!/bin/bash
# run_all.sh: the frozen protocol end to end (PROTOCOL.json): two bakes, validation evals + report, then the
# test period once (evals, baselines, report). Stage markers go to run_all.out.
cd ~/mskill
export CMF_GPU=cuda CMF_GPU_PROBE=0
BASE=~/skill-ode/cmf/qwen35-0.8b-q8_2f.cmf
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
say START
./bake.sh dir dir1 || exit 1
./bake.sh range range1 || exit 1
for t in dir range; do grep -E "=== bake|=== focused|checkpoint" runs/${t}1/bake.log | tail -4; done
[ -f runs/dir1/specialist.cmf ] && [ -f runs/range1/specialist.cmf ] || { say "NO SPECIALIST"; exit 1; }
say BAKES_DONE
mkdir -p results
for split in validation test; do
  for t in dir range; do
    python3 market_skill.py evaluate data/prep $t $split $BASE results/base_${t}_${split}.jsonl | tail -1
    python3 market_skill.py evaluate data/prep $t $split runs/${t}1/specialist.cmf results/skill_${t}_${split}.jsonl | tail -1
    python3 market_skill.py report data/prep $t $split results/base_${t}_${split}.jsonl results/skill_${t}_${split}.jsonl > results/report_${t}_${split}.json
    [ $split = test ] && python3 market_skill.py baselines data/prep $t test > results/baselines_${t}_test.json
  done
  say "${split^^}_DONE"
done
say ALL_DONE
