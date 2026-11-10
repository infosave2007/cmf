# fix_profiles.py DIR...: rewrite each <task>.profile from its <task>.phi windows (the saved profile started from the
# loaded generic profile, so its counts were the generic counts plus the task's); tokens = windows × 128.
import struct, sys, glob, os
import numpy as np
for d in sys.argv[1:]:
    for phi in sorted(glob.glob(f"{d}/*.phi")):
        name = os.path.basename(phi)[:-4]
        b = open(f"{d}/{name}.profile", "rb").read()
        nl, ne, _ = struct.unpack_from("<QQQ", b, 8)
        rows = np.fromfile(phi, dtype="<u4").reshape(-1, nl * ne)
        cnt = rows.sum(axis=0, dtype=np.uint64).astype(np.uint32)
        with open(f"{d}/{name}.profile", "wb") as f:
            f.write(b"CMFQ4PF\0" + struct.pack("<QQQ", nl, ne, rows.shape[0] * 128) + cnt.astype("<u4").tobytes())
        print(d, name, rows.shape[0], "windows", int(cnt.sum()), "picks")
