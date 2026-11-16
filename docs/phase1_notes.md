# Phase 1 Notes: Rough Pipeline on Real Clips

This is the "before" baseline for Phases 2–3. Every number comes from a single clip, so treat them as sanity checks, not results. The proper evaluations are planned for later phases: tracking (Phase 2), and shot and hit detection (Phase 3).

Pipeline: TrackNetV3 (pretrained, with InpaintNet), YOLOv8m-pose, 4 clicked court corners, rule-based hit detection, and the `unknown` shot placeholder. Config: `configs/pipeline.yaml`.

## Clips

| Clip | Source | Frames | Calibration |
| --- | --- | --- | --- |
| `Test/match1/1_05_03` | NYCU official test set (not in TrackNetV3's training data) | 395 @ 30 fps, 1280×720 | `configs/calibrations/nycu_test_match1_1_05_03.json` |
| Fuzhou Open 2019 final, set 1 rally 5 | ShuttleSet match video (YouTube), 17 s cut | 425 @ 25 fps, 1280×720 | `configs/calibrations/fuzhou2019_final_s1_r5.json` |
| `Amateur/match1/1_01_01` | NYCU amateur set | 189 @ 30 fps, 1280×720 | not possible (see below) |

## What works

- **Shuttle trail:** follows the shuttle. On the held-out test clip, TrackNetV3 is within 10 px of the label in 308 of the 314 frames where the shuttle is visible, with a median error of 2 px.
- **Players:** skeletons are drawn on exactly the two players in both broadcast clips. The umpire and line judges are filtered out by the court-area check; on the Fuzhou clip the umpire maps to x = −2.1 m. Near/far detection rates are 1.00/1.00 (test) and 0.86/0.98 (Fuzhou).
- **Court inset:** the dots stay in the correct halves and move with the players.
- **Hits on Fuzhou**, scored against ShuttleSet's labelled hit frames at ±5 frames (0.2 s):
  - 12 detected vs 16 true: precision 0.92, recall 0.69
  - hitter correct for 10 of 11 matched hits
  - missed hits at frames 120, 220, 320, 358 and 395
  - **These numbers are optimistic.** The `merge_same_hitter_s` rule was added after seeing failures on this exact clip; before it, only 6 hits were found and none after frame 229. The match is also in the ShuttleSet *val* split, so Phase 3 must report hit detection on the test matches only.

## What fails

1. **Hits at the net go to the wrong player.** When both players are at the net, the near player stands in front of the far player in the image. The near player's box (extended upward for the racket) then covers the shuttle, so far-player hits get credited to the near player. Example: test clip frame 225, and the run of four "near" hits from frame 139 to 260. Fix candidates: use wrist keypoints instead of boxes, or use the shuttle's direction after the hit (it travels away from the hitter).
2. **InpaintNet over-fills visibility.** Shuttle visible in 353 predicted frames vs 314 labelled on the test clip, and 100% of Fuzhou frames. It fills frames where the shuttle is truly out of view or not yet in play. Phase 2's comparison should report precision and recall separately so this shows up.
3. **Replay camera at the start of a clip.** The Fuzhou cut opens on about 2 s of a replay camera. That produced a false "serve" at frame 8 and made frame-0 calibration impossible. The app and `calibrate_court.py` now let you pick the calibration frame. Hits detected outside the main camera view are still a problem.
4. **NYCU amateur footage can't be calibrated.** All three amateur matches were filmed from a low camera behind or beside the court, the near corners are out of frame, and several courts share the hall. The 4-outer-corner calibration doesn't fit. Amateur clips are still usable for tracking evaluation, which needs no calibration.
   - TrackNetV3 alone on `Amateur/match1/1_01_01`: all 170 labelled visible frames within 10 px, median error 2.2 px, all 189 frames marked visible.
   - This number is probably optimistic: TrackNetV3's training data likely included these amateur matches.
   - Showing the full pipeline on amateur play needs either our own clips filmed from behind and above a baseline with the whole court in view, or calibration from any 4 known court points instead of the outer corners.
5. **Shot types are all `unknown`.** Expected until Phase 3.

## Runtime

| Clip | Device | End to end |
| --- | --- | --- |
| Fuzhou (425 frames) | RTX 3050 Laptop 4 GB, TrackNetV3 batch 2 | 134 s |
| Test 1_05_03 (395 frames) | CPU only (GPU unavailable to PyTorch in this session) | 1,222 s |

TrackNetV3 dominates the runtime. Its default batch size of 16 spills past 4 GB of GPU memory (474 s for the Fuzhou clip, vs 127 s at batch 2). It runs the temporal ensemble ("weight" mode), so each frame is predicted several times. If the GPU isn't available, the pipeline currently falls back to the CPU without saying so.
