"""Read the Nsight Systems trace of `timing.py 6 -5 0.005 profile` (six iterations as now, then
six overlapped), exported to SQLite:

    nsys profile -o ov5 --capture-range=cudaProfilerApi --capture-range-end=stop \
        -t cuda,nvtx,osrt --sample=cpu python timing.py 6 -5 0.005 profile
    nsys export --type sqlite -o ov5.sqlite ov5.nsys-rep
    python trace.py ov5.sqlite

For each iteration: the kernels' time on the GPU by the thread that launched them (the served
forward passes run as CUDA graphs, traced as whole graphs, so they are not in these sums), and
the time the main process's threads spent waiting in the OS: `pthread_cond_timedwait` is
where CPython's threads wait for the GIL, `pthread_cond_wait` the main thread waiting for the
autograd thread's backward pass, and `pthread_rwlock_wrlock` a lock between threads launching
kernels.
"""

import collections
import sqlite3
import sys


def union(intervals):
    total, start, end = 0, None, None
    for a, b in sorted(intervals):
        if end is None or a > end:
            if end is not None:
                total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    return total + (end - start if end is not None else 0)


def main(path):
    db = sqlite3.connect(path)
    names = dict(db.execute("select id, value from StringIds").fetchall())
    ranges = db.execute(
        "select start, end, text from NVTX_EVENTS where text in ('sync', 'overlap') order by start"
    ).fetchall()
    kernels = db.execute(
        "select k.start, k.end, r.globalTid from CUPTI_ACTIVITY_KIND_KERNEL k"
        " join CUPTI_ACTIVITY_KIND_RUNTIME r on k.correlationId = r.correlationId"
    ).fetchall()
    launching = {tid for *_, tid in kernels}
    waits = db.execute("select start, end, nameId, globalTid from OSRT_API").fetchall()
    main_thread = collections.Counter(tid for *_, tid in kernels).most_common(1)[0][0]
    for s, e, mode in ranges:
        inside = [(max(a, s), min(b, e), t) for a, b, t in kernels if b > s and a < e]
        by_thread = collections.defaultdict(list)
        for a, b, t in inside:
            by_thread[t].append((a, b))
        busy = union([(a, b) for a, b, _ in inside])
        print(
            f"{mode:8} wall {(e - s) / 1e9:.3f} s, kernels busy {busy / 1e9:.3f} s:",
            ", ".join(
                f"{'main' if t == main_thread else 'thread ' + str(t % 1000000)} "
                f"{len(v)} kernels {sum(b - a for a, b in v) / 1e9:.3f} s"
                for t, v in sorted(by_thread.items())
            ),
        )
        found = collections.defaultdict(lambda: [0, 0.0])
        for a, b, n, t in waits:
            if t in launching and b > s and a < e:
                key = ("main" if t == main_thread else f"thread {t % 1000000}", names.get(n, n))
                found[key][0] += 1
                found[key][1] += (min(b, e) - max(a, s)) / 1e9
        for (thread, call), (count, seconds) in sorted(found.items()):
            if seconds >= 0.01 and call.startswith("pthread"):
                print(f"          {thread:14} {call:24} {count:6} calls {seconds:.3f} s")


if __name__ == "__main__":
    main(sys.argv[1])
