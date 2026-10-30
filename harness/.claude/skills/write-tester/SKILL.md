---
name: write-tester
description: How to make a plugin measurable and how to write or change the concurrency tester (tools/bench/bench.py). Use when adding a plugin, tuning a model, or changing how the bench searches, measures or decides.
---

# Writing the tester

The bench answers one question: what is the largest batch worth running on this machine? Every rule below came from a real mistake, so keep them when changing it.

## Measure the real thing

- **Send exactly C requests together.** That is a true batch of C. A closed loop with C in flight doesn't measure a batch.
- **Use real inputs (`--inputs`).** VRAM per extra request grows with input size. A value tuned on small images overflowed on real ones.
- **Only judge work.** Start measuring when the plugin reports `generating`, never on a timer. A timer can't tell loading from a slow step, and it caused false failures.
- **Warm the model outside a step.** A reload isn't a slow step. Warm again after any cancel, because cancelling kills the worker.
- **Don't wait for tasks to end.** Token models get `--window` seconds (default 10) and stop by themselves through `limit_s`. Results are in tokens/s.

## Two flags, nothing else

| Flag | Meaning | Reference |
| --- | --- | --- |
| VRAM | Over 95 % of what was free at the start | Available, never the card's total: other programs count |
| Bandwidth | Memory bandwidth below the model's own best | 15 points below for the step, under half live for 3 s |

- **Don't use GPU utilization.** It reads high at low clock or power. An overflow showed 99 % utilization with 4 % bandwidth.
- **Don't use a fixed bandwidth target.** Each model has its own ceiling, so compare a model with itself.
- **Compare in points, not percent.** A healthy step went from 42 % to 35 %, which is a 17 % relative drop.
- **A tripped flag cancels the step at once.** Finishing it wastes time.

## Choose steps, don't sweep

High steps are the slowest, so they run only after a lower step has shown they could help.

1. C=1 is the baseline.
2. Mid-range (`--max` / 2). A short gap gives a poor VRAM slope.
3. Half of that. Compare it with mid-range:
   - It matches (within 5 %): the limit is below, so keep halving down.
   - Mid-range is better: halve upward, up to where VRAM is predicted to run out.
   - Mid-range tripped a flag: search below it.
4. Re-predict the VRAM ceiling after every good step, and never run a C predicted to exceed it.
5. Stop when the range is within one eighth, or at 1.

Pick the smallest C within 5 % of the best rate. Higher only adds latency and VRAM. A tripped step can't be chosen.

Save only to `tuning.yaml`, only after confirmation. Never edit the manifest.

## Make a plugin measurable

In the adapter:

```python
ctx.report_stage("generating")   # loading and preprocessing are over
ctx.tick(batch_size)             # once per generation step
```

`tick` counts tokens and raises `sdk.Stopped` at the time limit. Let it propagate: the runner turns it into a clean result, and the model stays loaded. If the engine has no step callback, call `tick` from a forward hook.

A plugin without `tick` still works: it runs whole batches and reports requests/s, judged on VRAM alone if it also doesn't report `generating`.

Also check the adapter's memory behaviour with a many-object input before tuning. Engines often batch internal work by default.

## Know what the number covers

Tokens/s covers generation only. If a task spends its time after generation, such as decoding masks, it can't see that cost. For those tasks, check latency on real inputs too.

## Test the search without a GPU

`choose_max` takes `measure(c, floor)`, so every decision can run on simulated curves in `tests/bench_test.py`. When you change the search:

1. Add a case for the behaviour, with the exact step order it should take.
2. Replay real numbers from the last run as a case.
3. Run `tests/bench_test.py`.
4. Only then run it on the GPU, and read every step of the output. A plausible final number can come from a wrong search.
