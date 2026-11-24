# show_report.py RESULTS SPLIT: base vs skill per market and target, with the baselines when present
import json, os, sys
res, split = sys.argv[1], sys.argv[2]
for t in ["dir", "range"]:
    p = f"{res}/report_{t}_{split}.json"
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    b = json.load(open(f"{res}/baselines_{t}_{split}.json")) if os.path.exists(f"{res}/baselines_{t}_{split}.json") else {}
    print(f"== {t} ({split})")
    for m, v in d.items():
        line = (f"  {m:8s} n {v['skill']['n']:4d} | base acc {v['base']['acc']*100:5.1f} bal {v['base']['bal_acc']*100:5.1f} | "
                f"skill acc {v['skill']['acc']*100:5.1f} bal {v['skill']['bal_acc']*100:5.1f} f1 {v['skill']['macro_f1']*100:5.1f} | "
                f"d acc {v['delta_acc']*100:+5.1f} CI [{v['delta_ci95'][0]*100:+.1f}, {v['delta_ci95'][1]*100:+.1f}] | "
                f"top20% {v['skill_top20_acc']*100:5.1f} | mix {v['skill_pred_mix']}")
        print(line)
        if m in b:
            print("           baselines: " + " | ".join(f"{k} {x['acc']*100:.1f}/{x['bal_acc']*100:.1f}" for k, x in b[m].items()))
