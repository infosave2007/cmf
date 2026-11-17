#!/bin/bash
# The fresh downloads as u16 shards with the 56M's tokenizer (gen1), next to its old code shard.
# Writes /root/sk/mix/{en2,ru2,math2}.u16 (+ .log); marker /root/sk/mix/shard.done
cd /root/sk && E=/root/sk/cmf/target-data/release/cortiq-embryo T=/root/sk/data/gen1.tokenizer.json M=/root/sk/mix
$E shard --tokenizer $T --inputs raw/fineweb-edu/001_00000.parquet raw/fineweb-edu/002_00000.parquet --out $M/en2.u16 > $M/en2.log 2>&1 &
$E shard --tokenizer $T --inputs raw/fineweb2-rus/000_00001.parquet --max-tokens 400000000 --out $M/ru2.u16 > $M/ru2.log 2>&1 &
$E shard --tokenizer $T --inputs raw/finemath/train-00001-of-00064.parquet --out $M/math2.u16 > $M/math2.log 2>&1 &
wait
ls -la $M; tail -n 2 $M/*.log
touch $M/shard.done
