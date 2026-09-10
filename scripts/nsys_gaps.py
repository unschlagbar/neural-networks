#!/usr/bin/env python3
"""Find the holes in an nsys capture: where the GPU ran no kernel, and why.

A timeline shows the holes but not their owner. This walks the kernel activity
of a report, unions it into busy/idle, buckets the idle by gap length, and for
the gaps that matter attributes them to the CUDA API call the host was inside
and to the kernel that ran once the wait ended.

    python3 scripts/nsys_gaps.py step100.nsys-rep [--min-gap-us 100] [--top 8]

Takes a .nsys-rep (converted once, next to it) or an already-converted .sqlite.
"""

import argparse
import collections
import os
import statistics
import subprocess
import sqlite3
import sys

BUCKETS = [
    (0, 5e3, "<5us"),
    (5e3, 20e3, "5-20us"),
    (20e3, 100e3, "20-100us"),
    (100e3, 1e6, "0.1-1ms"),
    (1e6, float("inf"), ">1ms"),
]


def to_sqlite(path):
    if path.endswith(".sqlite"):
        return path
    db = path.rsplit(".", 1)[0] + ".sqlite"
    if not os.path.exists(db) or os.path.getmtime(db) < os.path.getmtime(path):
        subprocess.run(
            ["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
             "-o", db, path],
            check=True, stdout=subprocess.DEVNULL,
        )
    return db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--min-gap-us", type=float, default=100.0,
                    help="gaps at least this long are attributed individually")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    con = sqlite3.connect(to_sqlite(args.report))
    q = lambda s: con.execute(s).fetchall()

    kernels = q("select start, end from CUPTI_ACTIVITY_KIND_KERNEL order by start")
    if not kernels:
        sys.exit("no kernel activity in this report")

    # Union the kernel intervals: concurrent kernels are one busy stretch, and
    # what is left between them is the idle this script is about.
    gaps = []
    busy = 0
    cur_start, cur_end = kernels[0]
    for s, e in kernels[1:]:
        if s > cur_end:
            gaps.append((cur_end, s))
            busy += cur_end - cur_start
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    busy += cur_end - cur_start
    span = kernels[-1][1] - kernels[0][0]
    t0, t1 = kernels[0][0], kernels[-1][1]

    # Only the overhead inside the capture distorts a measurement; the flushes
    # around it are the profiler's own bookkeeping.
    ov = q(f"""select coalesce(sum(min(end,{t1}) - max(start,{t0})), 0)
               from PROFILER_OVERHEAD where end > {t0} and start < {t1}""")[0][0]

    print(f"{len(kernels)} kernels over {span / 1e6:.1f} ms")
    print(f"  busy {busy / 1e6:.1f} ms ({100 * busy / span:.1f}%)  "
          f"idle {(span - busy) / 1e6:.1f} ms  "
          f"profiler overhead inside the capture {ov / 1e6:.2f} ms")

    lens = sorted(b - a for a, b in gaps)
    total = sum(lens)
    print(f"\n{len(lens)} gaps, median {statistics.median(lens) / 1e3:.1f} us")
    for lo, hi, label in BUCKETS:
        sel = [g for g in lens if lo <= g < hi]
        if sel:
            print(f"  {label:>9}: {len(sel):6d} gaps  {sum(sel) / 1e6:7.1f} ms "
                  f"({100 * sum(sel) / total:3.0f}% of idle)")

    big = [(a, b) for a, b in gaps if b - a >= args.min_gap_us * 1e3]
    if not big:
        return
    print(f"\n{len(big)} gaps >= {args.min_gap_us:.0f} us, "
          f"{sum(b - a for a, b in big) / 1e6:.1f} ms total")

    # What the host was doing while nothing ran, clipped to the gap: a call that
    # spans the whole hole is the one that owns it.
    calls = collections.Counter()
    time_in = collections.Counter()
    after = collections.Counter()
    for a, b in big:
        for name, n, t in q(f"""
                select s.value, count(*), sum(min(r.end,{b}) - max(r.start,{a}))
                from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id = r.nameId
                where r.start < {b} and r.end > {a} group by 1"""):
            calls[name] += n
            time_in[name] += t
        row = q(f"""select s.value from CUPTI_ACTIVITY_KIND_KERNEL k
                    join StringIds s on s.id = k.demangledName
                    where k.start >= {b} order by k.start limit 1""")
        if row:
            after[row[0][0]] += 1

    print("\n  host was inside:")
    for name, t in time_in.most_common(args.top):
        print(f"    {name:<28} {calls[name]:6d} calls  {t / 1e6:7.1f} ms")
    print("\n  kernel that ran once the gap ended:")
    for name, n in after.most_common(args.top):
        print(f"    {n:6d}  {name[:64]}")

    kinds = {1: "HtoD", 2: "DtoH", 8: "DtoD"}
    rows = q("""select copyKind, count(*), sum(end - start), sum(bytes)
                from CUPTI_ACTIVITY_KIND_MEMCPY group by 1 order by 3 desc""")
    if rows:
        print("\n  memcpy:")
        for kind, n, t, by in rows:
            rate = by / t * 1e9 / 1e9 if t else 0
            print(f"    {kinds.get(kind, kind):<5} {n:6d} copies  {t / 1e6:7.1f} ms  "
                  f"{by / 1e9:6.2f} GB  {rate:5.1f} GB/s")


if __name__ == "__main__":
    main()
