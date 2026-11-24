#!/bin/bash
# run_test.sh: the test period, once (PROTOCOL.json test_policy): base and frozen skills through serve, baselines, reports
cd ~/mskill
export CMF_GPU=cuda CMF_GPU_PROBE=0
BASE=~/skill-ode/cmf/qwen35-0.8b-q8_2f.cmf
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
for t in dir range; do
  python3 market_skill.py evaluate data/prep $t test $BASE results/base_${t}_test.jsonl | tail -1
  SERVE_ARGS="--skill mkt$t" python3 market_skill.py evaluate data/prep $t test runs/${t}1/overlay.cmf results/skill_${t}_test.jsonl | tail -1
  python3 market_skill.py report data/prep $t test results/base_${t}_test.jsonl results/skill_${t}_test.jsonl > results/report_${t}_test.json
  python3 market_skill.py baselines data/prep $t test > results/baselines_${t}_test.json
done
say TEST_DONE
