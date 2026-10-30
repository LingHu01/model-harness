# Bench notes

State of the concurrency tester, and what is left. Tuning was paused here so the models could be used.

## Current state

- Each function is tuned on its own, on one fixed input, so every request in a step is identical.
- Steps are whole tasks, 2 batches per step, in req/s, with tok/s shown alongside.
- Limits are saved per function in `tuning.yaml`. If functions mix in the queue, the lowest limit applies.
- 4B still uses an estimate (max 4, 1.6 GB per extra request). It has not been tuned with this bench.

## 2B, bag image, 2026-10-30

| Function | Max | req/s at max | p95 at max | tok/s at C=1 |
| --- | --- | --- | --- | --- |
| refer | 6 | 1.91 | 3.1 s | 20.0 |
| point | 8 | 1.67 | 4.8 s | 22.4 |
| detect | 8 | 1.30 | 6.2 s | 23.3 |
| count | 9 | 1.29 | 7.0 s | 23.3 |
| ocr | 12 | 0.70 | 17.6 s | 25.0 |
| segment | 6 | 0.43 | 13.9 s | 15.3 |
| matting | 6 | 0.43 | 13.9 s | 15.3 |

These fit this image only. Another size or object count can change them.

## What we learned

- Generation speed is about the same across tasks, 20 to 25 tok/s at C=1. Only the output length differs.
- Tok/s can't see work after generation. A 10 s token window recommended 14 for matting on 4 portraits, where whole tasks plateau at 4.
- For segment and matting, C=4 gives 0.42 req/s at 9.6 s against 0.43 req/s at 13.9 s for C=6, so 4 may suit flow better than the saved 6.

## To do

1. Speed up the bench: the full run took about 15 minutes.
   - Use 1 round per step. Identical inputs shouldn't need more.
   - Add a p95 budget of 3x C=1. A batch over budget stops itself through `limit_s`, with no reload. The mask decoder needs a stop check first.
2. Fix a false alarm: count at C=14 was cancelled at 65 % bandwidth, probably from the dip when generation ends. Stop counting bandwidth at that point.
3. Test the per-function and lowest-limit logic on real mixed requests.
4. Rerun the GPU self-test on the final code.
5. Decide flow and bulk limits. Proposal: pick whichever drains the queue fastest, `ceil(n / flow) * flow_time` against `ceil(n / bulk) * bulk_time`. Bulk equals flow if it isn't at least 10 % faster.
6. Count only unfinished requests in `tick`, which overstates tok/s in mixed-length batches.
7. Rename "tokens" to a generic unit, so non-text models fit the same tester.
8. Tune 4B.
9. Update `.claude/skills/write-tester/SKILL.md`, which still describes the older rules: `--window`, and comparing across a mix of inputs.
