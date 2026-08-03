#!/bin/bash
# The dsv4 toy gate. FIRST it proves the device came up: a WGSL parse error
# takes the whole module with it, wgpu falls back to the CPU with one WARN,
# and then every arm of the comparison agrees — vacuously.
S=/private/tmp/claude-501/-Users-oleg-Documents-cortiq-bot-cmfpublic/674db62a-643b-4641-b330-5feed6d40b67/scratchpad
E="CMF_GPU=wgpu CMF_SDOT=0 CMF_GPU_VRAM_MB=200 CMF_DSV4_GPU_LAYER=1 CMF_DSV4_GPU_ATTN=1 CMF_DSV4_GPU_MOE2=1 CMF_DSV4_SLOT_CHECK=1"

up=$(env $E RUST_LOG=info ./target/release/cortiq ppl $S/plain4.cmf --file $S/long.txt --tokens 4 2>&1 \
     | grep -ac "wgpu GPU path: on")
if [ "$up" != "1" ]; then
  echo "GATE ABORT: wgpu не поднялся — сравнение было бы холостым"
  env $E RUST_LOG=info ./target/release/cortiq ppl $S/plain4.cmf --file $S/long.txt --tokens 4 2>&1 \
    | grep -aiE "init failed|Shader|error" | head -4
  exit 1
fi
echo "устройство: поднялось"

# And that the chain really engages, which needs GPU_LAYER — without it both
# arms are the same path and agree for the wrong reason.
subs=$(env $E CMF_DSV4_CHAIN=1 CMF_DSV4_PROFILE=1 ./target/release/cortiq run $S/hx4.cmf \
       --prompt abc --max-tokens 12 2>&1 | grep -a "ЦЕПОЧКА" | head -1)
[ -z "$subs" ] && { echo "GATE ABORT: цепочка не включилась"; exit 1; }
echo "цепочка: $subs"

# The CPU arm. The chain and the layer frame run the SAME kernels, so a
# kernel that computes the wrong thing makes both of them agree — which is
# exactly how a rewritten projection passed this gate and then read 133.433
# where the CPU read 133.396. The host path is the only reference that is
# not also the thing under test.
echo "--- против CPU (стенды без индексатора обязаны совпасть) ---"
for toy in plain4 sc4; do
  cpu=$(env CMF_SDOT=0 CMF_GPU=off ./target/release/cortiq ppl $S/$toy.cmf --file $S/long.txt --tokens 120 2>/dev/null | grep -o "PPL = [0-9.]*")
  gpu=$(env $E CMF_DSV4_CHAIN=1 ./target/release/cortiq ppl $S/$toy.cmf --file $S/long.txt --tokens 120 2>/dev/null | grep -o "PPL = [0-9.]*")
  if [ "$cpu" != "$gpu" ]; then
    echo "$toy  CPU:$cpu  карта:$gpu  ← РАСХОЖДЕНИЕ С ХОСТОМ"
    exit 1
  fi
  echo "$toy  CPU:$cpu  карта:$gpu"
done

fail=0
for toy in hx4 plain4 sc4 q4 big2; do
  a=$(env $E CMF_DSV4_CHAIN=0 ./target/release/cortiq ppl $S/$toy.cmf --file $S/long.txt --tokens 120 2>/dev/null | grep -o "PPL = [0-9.]*")
  b=$(env $E CMF_DSV4_CHAIN=1 ./target/release/cortiq ppl $S/$toy.cmf --file $S/long.txt --tokens 120 2>/dev/null | grep -o "PPL = [0-9.]*")
  # plain/scored/hash stands must be EXACT; the two with indexers carry the
  # on-device top-k flip contract, so they get 0.05%.
  mark=""
  if [ "$a" != "$b" ]; then
    case $toy in
      q4|big2)
        d=$(python3 -c "import sys;x=float('$a'.split()[-1]);y=float('$b'.split()[-1]);print(abs(x-y)/x*100)")
        ok=$(python3 -c "print(1 if $d < 0.05 else 0)")
        if [ "$ok" = "1" ]; then mark="  (флип $d%)"; else mark="  ← РАСХОЖДЕНИЕ $d%"; fail=1; fi ;;
      *)
        d=$(python3 -c "x=float('$a'.split()[-1]);y=float('$b'.split()[-1]);print(abs(x-y)/x*100)")
        ok=$(python3 -c "print(1 if $d < 0.001 else 0)")
        # 0.001%: the split attention and the wide matvec sum in a different
        # lane order, so 'exact' now means 'below the float noise floor', not
        # 'the same bits'. Anything real is orders of magnitude above this.
        if [ "$ok" = "1" ]; then mark="  (шум $d%)"; else mark="  ← РАСХОЖДЕНИЕ $d%"; fail=1; fi ;;
    esac
  fi
  echo "$toy  кадр:$a  цепочка:$b$mark"
done
exit $fail
