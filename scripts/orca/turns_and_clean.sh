#!/bin/bash
# Five chat turns in one process: the default depth rule against a pinned depth 3, twice, in turn; then the pod's
# model files go (re-fetch: hf download infosave/OrcaCyber-Nano-1.0-Turbo-cmf, 14 s). turns_and_clean.sh LOG BINARY MODEL
cd /root/o; export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0
L=$1; B=$2; M=$3; : > $L
cat > /root/o/turns.txt <<'T'
Write a Rust function fn parse_kv(line: &str) -> Option<(String, String)> that parses "key = value", trims both sides, skips lines starting with #, with two unit tests.
Review this for security: uid = request.args.get("id"); db.execute("SELECT name FROM users WHERE id = " + uid). Give a fixed version.
Triage: 40 failed SSH logins for root from 203.0.113.45 in one minute, then one accepted publickey login for deploy from 192.0.2.10. What do you block first?
Explain to a new team member, in two short paragraphs, why defense in depth matters for a small company.
Объясни, чем отличается XSS от CSRF, и приведи по одному способу защиты от каждой атаки.
T
for rep in 1 2; do
  for arm in "default" "k3"; do
    E=""; [ $arm = k3 ] && E="CMF_GRAPH_SPEC_K=3 CMF_GRAPH_SPEC_PIN=1"
    env $E timeout 900 $B run $M --greedy --no-think --max-tokens 200 < /root/o/turns.txt > /root/o/turns_$arm$rep.out 2>/dev/null
    echo "$arm pass $rep: $(grep -oE 'decode [0-9.]+ tok/s' /root/o/turns_$arm$rep.out | awk '{print $2}' | tr '\n' ' ')" | tee -a $L
  done
done
rm -f /root/o/m/*.cmf /root/o/m/*.cmf.work; find / -xdev -name "*.cmf.work" 2>/dev/null | head; df -h / | tail -1 | tee -a $L
ps -eo pid,etime,cmd | grep -E "[c]ortiq|[p]ython3 .*(rebuild|passkey|hfup|dl)" | tee -a $L
echo TURNS_DONE >> $L
