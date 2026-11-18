"""Kernel time per training step from an nsys trace: the kernels between two AdamW launches
(emb_main_op3), the card's busy fraction, and the N heaviest kernels.

    python3 nsys_steps.py run/nsys-TAG.sqlite [N]
"""
import sqlite3,collections,sys
c=sqlite3.connect(sys.argv[1])
rows=c.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start").fetchall()
ad=[r for r in rows if r[2]=="emb_main_op3"]
print("adamw launches", len(ad))
lo=ad[2][1]; hi=ad[-1][1]; nsteps=len(ad)-3
tot=collections.defaultdict(lambda:[0,0]); busy=0; ce=None; cs=None
for s,e,n in rows:
    if s<lo or e>hi: continue
    tot[n][0]+=e-s; tot[n][1]+=1
    if ce is None or s>ce:
        if ce is not None: busy+=ce-cs
        cs,ce=s,e
    else: ce=max(ce,e)
busy+=ce-cs
print(f"steps {nsteps}: wall {(hi-lo)/1e6/nsteps:.1f} ms/step, busy {busy/1e6/nsteps:.1f}, util {busy/(hi-lo):.3f}")
allk=sum(v[0] for v in tot.values())
print(f"kernel sum {allk/1e6/nsteps:.1f} ms/step, launches {sum(v[1] for v in tot.values())/nsteps:.0f}/step")
top=int(sys.argv[2]) if len(sys.argv)>2 else 30
for n,(t,k) in sorted(tot.items(), key=lambda x:-x[1][0])[:top]:
    print(f"{n[:44]:<44} {t/1e6/nsteps:7.2f} ms  x{k/nsteps:6.0f}  avg {t/k/1e3:7.1f} us")
