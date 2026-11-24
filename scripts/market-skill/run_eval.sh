#!/bin/bash
# run_eval.sh: after the two frozen bakes — skill export + overlay (the CUDA-served form), a held-set parity check
# against the bake's own held accuracy, validation evals + report, then the test period once.
cd ~/mskill
export CMF_GPU=cuda CMF_GPU_PROBE=0
B=~/cmf-skillode/target/release/cortiq
BASE=~/skill-ode/cmf/qwen35-0.8b-q8_2f.cmf
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p results
for t in dir range; do
  $B skill export runs/${t}1/specialist.cmf --base $BASE --id mkt$t -o runs/${t}1/mkt$t.skill.cmf 2>&1 | grep -v INFO | tail -1
  rm -f runs/${t}1/overlay.cmf
  $B skill apply $BASE runs/${t}1/mkt$t.skill.cmf --overlay -o runs/${t}1/overlay.cmf 2>&1 | grep -v INFO | tail -1
done
say OVERLAYS_DONE
for t in dir range; do
  SERVE_ARGS="--skill mkt$t" python3 market_skill.py evaluate data/prep $t held runs/${t}1/overlay.cmf results/skill_${t}_held.jsonl | tail -1
  grep -m3 -E "hybrid graph|skill" results/skill_${t}_held.jsonl.serve.log | cut -c1-200
  python3 -c "
import json; r=[json.loads(l) for l in open('results/skill_${t}_held.jsonl')]
print('held parity ${t}: serve overlay acc %.2f%% on %d (the bake reported its held accuracy above)' % (100*sum(x['pred']==x['label'] for x in r)/len(r), len(r)))"
  grep "=== focused" runs/${t}1/bake.log | cut -c1-200
done
say HELD_DONE
for split in validation test; do
  for t in dir range; do
    [ -s results/base_${t}_${split}.jsonl ] || python3 market_skill.py evaluate data/prep $t $split $BASE results/base_${t}_${split}.jsonl | tail -1
    SERVE_ARGS="--skill mkt$t" python3 market_skill.py evaluate data/prep $t $split runs/${t}1/overlay.cmf results/skill_${t}_${split}.jsonl | tail -1
    python3 market_skill.py report data/prep $t $split results/base_${t}_${split}.jsonl results/skill_${t}_${split}.jsonl > results/report_${t}_${split}.json
    [ $split = test ] && python3 market_skill.py baselines data/prep $t test > results/baselines_${t}_test.json
  done
  say "${split^^}_DONE"
  [ $split = validation ] && [ -n "$STOP_AFTER_VALIDATION" ] && exit 0
done
say ALL_DONE
