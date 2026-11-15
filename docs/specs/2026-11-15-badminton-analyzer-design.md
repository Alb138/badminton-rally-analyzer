# Badminton Rally Analyzer: Design

## Context
- Solo portfolio project for Data Scientist internship applications. User is fairly advanced in Python/ML.
- The topic came from an earlier shortlist (late Sept 2026). This design adapts that team proposal to a solo build with no hard deadline.
- The project must give recruiters two things:
  1. a demo they can watch: rally video in, annotated video out;
  2. evidence of careful experimentation: two head-to-head model comparisons with honest evaluation that doesn't leak test data.
- The user must be able to explain every part in an interview.
- Working folder: `C:\Users\Albert Mario Laurent\Projects\Project 2`, currently empty.

## Goals
- **Input:** a single singles rally clip, broadcast-style view from behind or above the baseline.
- **Output:** an annotated MP4 showing:
  - the shuttle's path
  - player skeletons
  - a label on each shot
  - an inset top-down court with player positions
- **Stats panel:** rally length, shot counts by type, hit locations, a movement heatmap and distance covered for each player.
- **Comparison 1, shuttle tracking:** single-frame YOLOv8 detector vs. a multi-frame TrackNet-style heatmap model.
  - Target F1 ≥ 0.85 on broadcast test videos.
  - Report amateur-footage F1 separately.
- **Comparison 2, shot recognition:** pose-sequence model vs. raw-video-clip model.
  - Target macro-F1 ≥ 0.70 on the class list fixed in Phase 0.
- **Public output:**
  - a GitHub repo whose README carries results tables, an architecture diagram and limitations
  - a hosted demo
  - a short write-up

## Non-goals
- Doubles
- 3D trajectory reconstruction
- Real-time or live video
- Splitting a full match into rallies (input is one rally; this is a possible later stretch)
- Automatic court-line detection (four clicked corners are enough)
- Tactics or next-shot prediction
- A mobile app

## Anti-goals (failure even if it "works")
- Leaky evaluation, such as frames from one match in both train and test. All splits are by match.
- Numbers that can't be reproduced. Seeds, configs and data splits are committed.
- A demo that only works on training videos. Showcase clips come from held-out or amateur footage.
- Presenting other people's work as the user's own. Pretrained parts are credited. The user's own models (YOLOv8 fine-tune, TrackNet-style model, both shot classifiers) are clearly identified as their work.
- A project that never comes together. The end-to-end pipeline must keep running after every phase.

## Constraints
- **Team and deadline:** solo, no deadline.
- **Compute (my assumption):** free cloud GPUs on Colab/Kaggle, with the local Windows machine used for development and inference. Code is a Python package plus scripts, with thin notebooks that call it from Colab.
- **NYCU shuttlecock dataset:**
  - 26 videos, about 78k frames, 720p at 30 fps
  - 23 professional videos and 3 amateur
  - per-frame CSV with Visibility, X and Y
- **ShuttleSet (KDD 2023):**
  - 44 matches, 36,492 strokes
  - stroke-level annotations: shot type, hit frame, positions
  - the source videos must be fetched separately, and whether they're still available is unverified
- **Licensing:** research-use datasets. The demo hosts only short clips with credit and never re-hosts whole YouTube matches.

## Approaches considered
- **A. Rough end-to-end first, then upgrade each part through the comparisons. Chosen.**
  - Phase 1 wires pretrained parts into a working video-in, annotated-video-out pipeline.
  - Each later comparison swaps its winner into the pipeline.
  - Why: there's always something demoable, integration bugs surface early, and each experiment visibly improves the output, which makes a good interview story.
- **B. Build each component in depth, integrate at the end. Rejected.** Nothing to show until late, and the hardest integration surprises come last.
- **Earlier directions, set aside by the user:** data-only tactics analysis on ShuttleSet; a BWF match-result predictor.

## Pipeline
```
rally.mp4
  ├─ shuttle tracker ────────► shuttle (x, y, visible) per frame
  ├─ YOLOv8-pose ────────────► keypoints for the 2 on-court players
  │                            (filtered by court polygon)
  ├─ court homography ───────► player feet + hit points in court coordinates
  │   (4 clicked corners)
  ├─ hit detection ──────────► hit frames
  │   (shuttle direction change + nearest player)
  ├─ shot classifier ────────► shot type per hit
  │   (window of frames around each hit)
  ├─ stats ──────────────────► rally summary, heatmaps (JSON)
  └─ renderer ───────────────► annotated.mp4
```
- **Data contract:** a per-frame table (frame, shuttle, players, court coordinates) and a per-shot table (hit frame, hitter, shot type, court position). Each stage reads and writes these tables, so a stage can be swapped without touching the others.
- **Hit detection** wasn't in the original proposal, but shot recognition needs it at inference time.
  - Phase 1 uses a simple rule: shuttle direction change near a player.
  - It is evaluated against ShuttleSet's annotated hit frames.

## Phases (each ends with the pipeline still running)
0. **Setup and data audit**
   - Create the repo and environment. Download the NYCU dataset and the ShuttleSet annotations.
   - Count how many of the 44 ShuttleSet videos can still be fetched.
   - Fix the shot class list by merging rare types, based on class counts.
   - Define match-level train/val/test splits.
   - Set aside an amateur test set: the 3 NYCU amateur videos, plus optional clips the user films.
1. **Rough end-to-end version**
   - Parts: public TrackNetV3 weights, pretrained YOLOv8-pose, clicked corners, rule-based hit detection, a placeholder shot classifier (pose features + gradient boosting, or "unknown"), the renderer, and a local Gradio app.
   - Exit: any rally clip produces an annotated video.
2. **Tracking comparison**
   - Model 1: fine-tune YOLOv8 on small boxes built from the point labels.
   - Model 2: the user's own TrackNet-style model (consecutive frames in, heatmap out), implemented from the paper.
   - Both use the same split and the TrackNet paper's distance-tolerance convention, so results line up with published numbers.
   - Report precision, recall, F1 and FPS, separately for broadcast and amateur footage.
   - Swap the winner into the pipeline.
3. **Shot-recognition comparison**
   - Build stroke clips from ShuttleSet videos and hit frames.
   - Pose model: hitter keypoint sequence into a temporal model (ST-GCN or GRU/transformer).
   - Video model: fine-tune a pretrained video backbone (X3D or VideoMAE) on clips cropped to the hitter.
   - Report macro-F1, per-class F1, a confusion matrix and inference cost.
   - Swap the winner into the pipeline.
4. **Court analytics:** heatmaps, distance covered, rally length, shot distribution, hit-position maps, per-rally report.
5. **App and publish**
   - Gradio app on Hugging Face Spaces. The free CPU tier is slow, so it ships precomputed example clips and limits uploads to short clips.
   - README with results tables and limitations, a short write-up, and a demo GIF.

## Calls I made (user can override)
- Python and PyTorch; Ultralytics YOLOv8 and YOLOv8-pose; OpenCV for the homography; Gradio on Hugging Face Spaces; Colab/Kaggle GPUs.
- Singles only. Input is a single rally clip in a broadcast-style view.
- Manual 4-corner court calibration.
- Hit detection is its own component. The hitter is the player nearest the shuttle at the hit frame.
- The TrackNet-style model is implemented by the user. TrackNetV3's public weights are used only in Phase 1 as a stand-in.
- All evaluation splits are by match.

## Left to the builder
- Model architectures and hyperparameters, and the exact merged class list (after the Phase 0 audit).
- Repo layout and the experiment tracker (MLflow or W&B).
- Frame-window sizes for the shot clips.

## Risk to resolve in Phase 0
- Both shot models need the ShuttleSet match videos.
- If too few can be fetched for a match-level split (roughly fewer than 25), stop and pick a fallback before Phase 3. Options:
  - hand-label strokes on the NYCU videos
  - another stroke dataset
  - narrower classes

## Verification
- **Phase 1:** run the pipeline CLI on a held-out NYCU clip and an amateur clip. Each must produce an MP4 with the shuttle trail, skeletons and court inset, plus a stats JSON. Open both and check them by eye.
- **Phase 2:** evaluation script prints precision, recall and F1 for both trackers on the match-level test split, broadcast and amateur separately. Target ≥ 0.85 broadcast F1 for the better model.
- **Phase 3:** evaluation script prints macro-F1, per-class F1 and a confusion matrix for both shot models on held-out matches. Target ≥ 0.70 macro-F1.
- **Hit detection:** precision and recall against ShuttleSet hit frames, within a small frame tolerance.
- **Phase 5:** the hosted Space loads, plays the precomputed examples and processes a short uploaded clip. A fresh clone plus the README steps reproduces the reported numbers from saved configs and seeds.
