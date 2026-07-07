#!/usr/bin/env bash
# ДЕМО РОЯ: один .cmf-файл — backbone + несколько скиллов + роутинг по
# физике сигнала (recon-argmin), без каких-либо аргументов «какой скилл
# брать». То, чего нет ни у GGUF, ни у safetensors (Patent 15).
#
# Требует: models/qwen35-swarm.cmf (этап B2) и собранный cortiq.
# Запуск:  ./demo/swarm_demo.sh [ru-текст-для-ppl]

set -euo pipefail
cd "$(dirname "$0")/.."

SWARM="${SWARM:-models/qwen35-swarm.cmf}"
RU_TEXT="${1:-../NVG_NEURAL_PRACTICE_RU.md}"
CLI=./target/release/cortiq
export RUST_LOG=error
[ -f "$SWARM" ] || { echo "нет $SWARM — соберите рой (make_skill + --skills)"; exit 1; }
[ -x "$CLI" ] || cargo build --release -q

hr() { printf '\n\033[1m── %s\033[0m\n' "$*"; }

hr "1. Один файл — вся экосистема (реестр скиллов живёт в header)"
$CLI info "$SWARM" | grep -E "Model|Arch|Params|Tensors|Quant"
python3 - "$SWARM" <<'EOF'
import sys
sys.path.insert(0, "python")
from cmf_reader import CmfReader
r = CmfReader(sys.argv[1])
total = sum(e["nbytes"] for e, _ in r.tensors.values())
for s in r.skills:
    sk = sum(e["nbytes"] for n, (e, _) in r.tensors.items()
             if n.startswith(f"skill.{s['id']}."))
    q = s.get("quality") or {}
    print(f"  skill '{s['id']}': {sk/1e6:.0f} МБ ({sk/total*100:.1f}% файла)"
          + (f" | контракт качества: {q.get('metric')} "
             f"{q.get('backbone')}→{q.get('overlaid')}" if q else ""))
EOF

hr "2. Роутинг без подсказок: recon-argmin по φ-подпространствам (P1)"
echo '  промпт: "Объясни, как работает нейронная сеть" (русский)'
$CLI route "$SWARM" -p "Объясни, как работает нейронная сеть" 2>/dev/null | sed 's/^/  /'
echo '  промпт: "fn main() { let mut v: Vec<u32> = ... }" (Rust-код)'
$CLI route "$SWARM" -p "fn main() { let mut v: Vec<u32> = vec![]; }" 2>/dev/null | sed 's/^/  /'

hr "3. Скилл читается ВМЕСТО тензора бэкбона (замещение, не сложение)"
python3 - "$SWARM" <<'EOF'
import sys
sys.path.insert(0, "python")
from cmf_reader import CmfReader
r = CmfReader(sys.argv[1])
name = next(n for n in r.tensors if n.startswith("skill.ru."))
orig = name[len("skill.ru."):]
import numpy as np
d = float(np.abs(r.tensor(orig) - r.tensor(orig, skill="ru")).max())
print(f"  {orig}:")
print(f"    backbone и skill.ru — разные веса (max|Δ| = {d:.4f}), один mmap, ноль копий")
EOF

hr "4. Качество вживую: PPL русского текста, backbone vs скилл"
if [ -f "$RU_TEXT" ]; then
    B=$($CLI ppl "$SWARM" --file "$RU_TEXT" --tokens 256 2>/dev/null | grep -o 'PPL = [0-9.]*' | cut -d' ' -f3)
    S=$($CLI ppl "$SWARM" --skill ru --file "$RU_TEXT" --tokens 256 2>/dev/null | grep -o 'PPL = [0-9.]*' | cut -d' ' -f3)
    A=$($CLI ppl "$SWARM" --blend auto --file "$RU_TEXT" --tokens 256 2>/dev/null | grep -o 'PPL = [0-9.]*' | cut -d' ' -f3)
    echo "  backbone:        PPL $B"
    echo "  --skill ru:      PPL $S"
    echo "  --blend auto:    PPL $A   (роутер сам выбрал смесь, клейм 14)"
else
    echo "  (текст $RU_TEXT не найден — шаг пропущен)"
fi

hr "5. Генерация с автоблендом (роутер выбирает скиллы сам)"
$CLI run "$SWARM" -p "Кратко: что такое нейрон?" --greedy --blend auto 2>/dev/null \
    | grep -v "Loading\|Ready:" | head -12 | sed 's/^/  /'

hr "6. ЖИВОЙ ТРЕЙС: φ-когерентность (E) и переключения скилла пер-токенно"
echo '  промпт: Rust-код → скилл переключается по ходу; видно в колонке E и ▸'
CMF_MAX_SEQ=200 CMF_ROUTE_PERIOD=4 \
    $CLI run "$SWARM" --route-dynamic --greedy --trace \
    -p "fn main() { let mut total: u64 = 0; for i in 0..100 { total += i; } }" 2>/dev/null \
    | sed -n '/трейс/,$p' | head -26 | sed 's/^/  /'
echo '  (conf = Born-масса уверенности; E = recon-когерентность ‖r−BBᵀr‖²/‖φ‖²;'
echo '   ▸ = переключение через tensor-source indirection на лету, ноль копий;'
echo '   два порога e_on<e_off гасят дребезг — VMF переход первого рода)'

hr "7. ИНТРОСПЕКЦИЯ БЕЗ ГЕНЕРАЦИИ: cortiq explain (какой скилл и почему)"
$CLI explain "$SWARM" -p "Напиши функцию на Rust" --top 5 2>/dev/null \
    | grep -vE "Loading|Opened|explain:|Промпт" | sed 's/^/  /'

printf '\n\033[1mИтог: один файл, N навыков, селекция по физике сигнала, каждый скилл\nс измеренным контрактом качества, роутинг ЖИВОЙ (пер-токенный) и\nНАБЛЮДАЕМЫЙ (--trace, explain). Хранение = бэкбон + Σ дельт.\033[0m\n'
