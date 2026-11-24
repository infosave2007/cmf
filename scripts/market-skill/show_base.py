import json, sys
for t in ["dir", "range"]:
    d = json.load(open(f"{sys.argv[1]}/baselines_{t}_{sys.argv[2]}.json"))
    print("==", t)
    for m, rows in d.items():
        print(" ", m, " | ".join(f"{k}: acc {v['acc']*100:.1f} bal {v['bal_acc']*100:.1f}" for k, v in rows.items()))
