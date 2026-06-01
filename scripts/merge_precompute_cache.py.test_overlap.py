"""Quick overlap detection test for shard 0 on bjb2-40593."""
import re
from collections import defaultdict
from pathlib import Path

PARTIAL_RE = re.compile(r"^partial_s(\d+)_(\d+)_(\d+)\.pt$")
cache_dir = Path("/root/autodl-tmp/crm_cache_pi05/object")

by_start = defaultdict(list)
all_items = []
for f in cache_dir.iterdir():
    m = PARTIAL_RE.match(f.name)
    if not m or int(m.group(1)) != 0:
        continue
    s, e = int(m.group(2)), int(m.group(3))
    by_start[s].append((e, f.name))
    all_items.append((s, e, f.name))

print(f"Shard 0: {len(all_items)} files")
dup_starts = sorted({s for s in by_start if len(by_start[s]) > 1})
print(f"Duplicate start positions: {dup_starts}")
for s in dup_starts:
    print(f"  start={s}: {by_start[s]}")

# Detect interval overlap (not just same-start)
all_items.sort(key=lambda x: x[0])
overlaps = []
for i in range(len(all_items)):
    for j in range(i+1, len(all_items)):
        s1, e1, n1 = all_items[i]
        s2, e2, n2 = all_items[j]
        if s2 < e1 and not (s1 == s2 and e1 == e2):
            overlaps.append((n1, n2))
            break
print(f"\nInterval overlaps detected: {len(overlaps)}")
for n1, n2 in overlaps[:10]:
    print(f"  {n1} <-> {n2}")
