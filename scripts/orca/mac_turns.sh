#!/bin/bash
# Five chat turns in one process on native Metal with drafts (the trial change), after mac_measure.sh. mac_turns.sh LOG
R=/Users/oleg/dev/orca-runs/mac; B=/Users/oleg/dev/cmf-orca/target/release/cortiq; M=$R/orcacyber-nano-1.0-turbo.cmf; L=$1; : > $L
until grep -q MAC_DONE $R/mac.log 2>/dev/null; do sleep 20; done
cat > $R/turns.txt <<'T'
Write a Rust function fn parse_kv(line: &str) -> Option<(String, String)> that parses "key = value", trims both sides, skips lines starting with #, with two unit tests.
Review this for security: uid = request.args.get("id"); db.execute("SELECT name FROM users WHERE id = " + uid). Give a fixed version.
Triage: 40 failed SSH logins for root from 203.0.113.45 in one minute, then one accepted publickey login for deploy from 192.0.2.10. What do you block first?
Explain to a new team member, in two short paragraphs, why defense in depth matters for a small company.
Объясни, чем отличается XSS от CSRF, и приведи по одному способу защиты от каждой атаки.
T
sleep 30
CMF_GRAPH_SPEC=1 RUST_LOG=info $B run $M --greedy --no-think --max-tokens 200 < $R/turns.txt > $R/turns.out 2> $R/turns.err
echo "turns drafts: $(grep -oE 'decode [0-9.]+ tok/s' $R/turns.out | awk '{print $2}' | tr '\n' ' ')" | tee -a $L
sed -e 's/\x1b\[[0-9;]*m//g' $R/turns.err | grep -E "speculation" | sed -e 's/^.*pipeline: //' | cut -c1-150 >> $L
echo TURNS_DONE >> $L
