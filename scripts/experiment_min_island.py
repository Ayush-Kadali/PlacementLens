#!/usr/bin/env python3
"""Experiment: is a "skip small device islands" rule as good as trying everything?

Real runtimes avoid tiny accelerator chunks with a rule of thumb (for example
Torch-TensorRT's min_block_size). We try the same idea on our verification
matrix: start from maximal placement, move every device island shorter than k
nodes back to the CPU, and compare the modeled cost with the exhaustive optimum.

Run scripts/check_matrix.py first. Costs are the illustrative model, not hardware.
"""
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/verification")
reports = sorted(p for p in out.glob("*.json") if p.name != "summary.json")
if not reports:
    sys.exit("no reports found; run scripts/check_matrix.py first")


def drop_small_islands(placement, k):
    return re.sub(r"A+", lambda m: m.group() if len(m.group()) >= k else "C" * len(m.group()), placement)


rows = {k: {"match": 0, "infeasible": 0, "excess": []} for k in (1, 2, 3)}
worst = {}
for path in reports:
    plans = {r["placement"]: r["modeled_ms"] for r in json.loads(path.read_text())["results"]}
    best = min(plans.values())
    width = len(next(iter(plans)))
    # A node counts as device-capable if any feasible plan puts it on the device.
    maximal = "".join("A" if any(p[i] == "A" for p in plans) else "C" for i in range(width))
    for k, row in rows.items():
        chosen = drop_small_islands(maximal, k)
        if chosen not in plans:
            row["infeasible"] += 1
            continue
        excess = plans[chosen] - best
        row["excess"].append(excess)
        if excess < 1e-9:
            row["match"] += 1
        elif excess > worst.get(k, (0, ""))[0]:
            worst[k] = (excess, f"{path.stem}: chose {chosen}, best was {min(plans, key=plans.get)}")

print(f"{len(reports)} cases\n")
print("min island k | matches optimum | infeasible | mean excess ms | worst excess ms")
for k, row in rows.items():
    ex = row["excess"]
    mean = sum(ex) / len(ex) if ex else 0.0
    print(f"{k:12d} | {row['match']:15d} | {row['infeasible']:10d} | {mean:14.4f} | {max(ex, default=0):15.4f}")
print()
for k, (excess, where) in sorted(worst.items()):
    print(f"k={k} worst case: +{excess:.4f} ms in {where}")
