import sys
print("=== D5 condition values test ===", flush=True)
try:
    from dataset.multigame.handlers.d5_handler import D5Handler
    from collections import Counter
    h = D5Handler()
    print(repr(h), flush=True)
    cond_dist = {}  # {enum: Counter of condition values}
    for s in h:
        re = s.meta["reward_enum"]
        cv = list(s.meta["conditions"].values())[0]
        cond_dist.setdefault(re, Counter())[cv] += 1
    for re in sorted(cond_dist):
        vals = sorted(cond_dist[re].items())
        n_unique = len(vals)
        fn = "collectable" if re == 4 else ""
        print(f"  enum={re} {fn}: {n_unique} unique values → {vals}", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
print("=== Done ===", flush=True)

