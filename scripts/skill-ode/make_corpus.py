#!/usr/bin/env python3
"""make_corpus.py: the ode_repro.py training problems (seed 0) as cortiq skill-bake text files.
train.txt = the same 300 problems the PyTorch run trains on; held.txt = 40 more from the TRAIN coefficients."""
import random, importlib.util, sys
spec = importlib.util.spec_from_file_location("r", "ode_repro.py"); r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
train = r.make_set(r.TRAIN, 300, random.Random(0))
held = r.make_set(r.TRAIN, 40, random.Random(7))
for name, items in [("train.txt", train), ("held.txt", held)]:
    with open(name, "w") as f:
        for _, p, s, _ in items:
            f.write(p + " " + s + "\n\n")
print(len(train), len(held))
