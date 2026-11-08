#!/bin/bash
# Build the branch binary, import the GGUF to q4tp (with Qwen3.8-27B's tokenizer.json), run the two unit tests.
cd /root/o
. ~/.cargo/env
export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0
L=/root/o/import.log; : > $L
(cd src && t0=$(date +%s) && cargo build --release -p cortiq-cli > /root/o/build3.log 2>&1; echo "build3 rc=$? $(( $(date +%s)-t0 )) s" >> $L)
B=/root/o/src/target/release/cortiq
mkdir -p /root/o/tokdir && cp /root/o/qtok/tokenizer.json /root/o/tokdir/ && cp /root/o/qtok/tokenizer_config.json /root/o/tokdir/
# The GGUF's own chat template (Qwen's minus the generation markers) goes in as the template.
python3 - <<'PY' >> $L 2>&1
import sys
sys.argv=["x"]
from gguf import GGUFReader
r=GGUFReader("/root/o/m/OrcaCyber-Nano-1.0-AI-Defense-Mix-15.7G.gguf")
f=r.fields["tokenizer.chat_template"]
ct=bytes(f.parts[f.data[0]]).decode()
open("/root/o/tokdir/chat_template.jinja","w").write(ct); print("template", len(ct))
PY
t0=$(date +%s)
/usr/bin/time -v $B import-gguf /root/o/m/OrcaCyber-Nano-1.0-AI-Defense-Mix-15.7G.gguf --quant q4tp --output /root/o/m/orca-q4tp.cmf --tokenizer-dir /root/o/tokdir > /root/o/imp.out 2> /root/o/imp.err
echo "import rc=$? $(( $(date +%s)-t0 )) s" >> $L
grep -a "MTP head\|note\|Maximum resident\|Elapsed" /root/o/imp.err | tail -6 >> $L
ls -l /root/o/m/orca-q4tp.cmf >> $L 2>&1
(cd src && cargo test --release -p cortiq-cli --bin cortiq -- qwen35_gguf_mtp iq3s_matches > /root/o/test3.log 2>&1; echo "test rc=$?" >> $L; grep -a "test result\|^test " /root/o/test3.log >> $L)
echo IMPORT_DONE >> $L
