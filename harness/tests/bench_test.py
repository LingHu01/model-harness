"""The bench's search against simulated models: it must stop at the right C without sweeping every step."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))

import bench  # noqa: E402

failed = []


def model(base, per_extra, tput, bw, oom_at=None):
    """tput(c) and bw(c) are callables; VRAM grows linearly."""
    ran = []

    def measure(c, floor):
        ran.append(c)
        err = ["oom"] if oom_at and c >= oom_at else []
        return {"c": c, "tput": tput(c), "p95": c * 1.0, "vram": base + (c - 1) * per_extra, "bandwidth": bw(c),
                "k": c, "rounds": 2, "errors": err, "spilled": False, "abort": None}
    return measure, ran


def expect(name, measure, ran, want, cmax=16, limit=15.0, most=5, never_above=None):
    best, _ = bench.choose_max(measure, cmax, limit, say=lambda *_: None)
    ok = best == want and len(ran) <= most and (never_above is None or max(ran) <= never_above)
    print(("PASS " if ok else "FAIL ") + f"{name}: chose {best} (want {want}), ran {ran}")
    if not ok:
        failed.append(name)


def linear(c):
    return 0.4 * c


def saturating(at):
    return lambda c: 0.4 * min(c, at)


FLAT = lambda c: 40  # noqa: E731

# no flag and throughput keeps rising: halves upward, and stops within an eighth of --max
m, r = model(5, 0.2, linear, FLAT)
expect("no flag trips", m, r, 15, most=6)

# VRAM decides: C=11 is 15 GB, so 12 and above must never run
m, r = model(5, 1.0, linear, FLAT)
expect("VRAM limit", m, r, 11, most=6, never_above=11)

# only the free VRAM counts: ComfyUI holding memory lowers the ceiling
m, r = model(5, 1.0, linear, FLAT)
expect("less VRAM available", m, r, 5, limit=9.0, most=4)

# bandwidth decides: from C=7 the GPU can't keep up, today's overflow signature
m, r = model(5, 0.1, linear, lambda c: 40 if c < 7 else 10)
expect("bandwidth drop", m, r, 6, most=6)

# a normal dip between steps is not a flag: real run, 42 % at C=1 and 35 % at C=2
m, r = model(5, 0.1, linear, lambda c: 42 if c == 1 else 35)
expect("small bandwidth dip", m, r, 15, most=6)

# the final real 2B run: the search must find 4 without ever running 16, the step that cost the most
real = {1: (0.43, 40), 2: (0.52, 42), 3: (0.59, 38), 4: (0.62, 41), 8: (0.65, 43), 16: (0.65, 41)}
m, r = model(8.36, 0.27, lambda c: real[c][0], lambda c: real[c][1])
expect("real 2B numbers", m, r, 4, limit=13.39, most=5, never_above=8)
check_order = r == [1, 8, 4, 2, 3]
print(("PASS " if check_order else "FAIL ") + f"real 2B order: {r}")
failed += [] if check_order else ["real 2B order"]

# 8 is better than 4, so it must go up to 12 next
m, r = model(5, 0.1, saturating(11), FLAT)
expect("limit above mid-range", m, r, 11, most=7)
check_order = r[:4] == [1, 8, 4, 12]
print(("PASS " if check_order else "FAIL ") + f"goes up after 4: {r}")
failed += [] if check_order else ["goes up after 4"]

# 8 beats 4, but 12 adds nothing: the limit is between 4 and 8
m, r = model(5, 0.1, saturating(6), FLAT)
expect("limit between 4 and 8", m, r, 6, most=6)

# plateau with no flag: 0.64 to 0.66 req/s from C=4 to C=16
m, r = model(5, 0.1, lambda c: min(0.65, 0.30 + 0.09 * c), FLAT)
expect("throughput plateau", m, r, 4, most=5, never_above=8)

# bandwidth collapsed from C=2: found from above, because the jump cancels in seconds
m, r = model(5, 0.1, linear, lambda c: 40 if c == 1 else 5)
expect("flag at C=2", m, r, 1, most=4)

# one request does not fit
m, r = model(20, 0.1, linear, FLAT)
expect("model too big", m, r, 0, most=1)

# prediction was wrong: real OOM at 8 although VRAM looked fine
m, r = model(5, 0.1, linear, FLAT, oom_at=8)
expect("OOM below the prediction", m, r, 7, most=5)

print("\nall passed" if not failed else f"\nfailed: {failed}")
sys.exit(1 if failed else 0)
