# Badminton Rally Analyzer

Upload a singles rally clip and get back an annotated video showing the shuttle's path, the players' skeletons, a label on each shot and a top-down court view, plus rally stats like rally length, shot counts and distance covered. Behind the demo are two model comparisons: a single-frame detector against a multi-frame tracker for the shuttle, and a pose-based model against a video-based model for recognizing shots.

Status: Phase 1 in progress.

## Credits

- **NYCU Shuttlecock Trajectory Dataset**: per-frame shuttle labels used for tracking ([dataset page](https://hackmd.io/@TUIK/rJkRW54cU)).
- **ShuttleSet** (Wang et al., KDD 2023): stroke-level annotations of professional singles matches ([paper](https://arxiv.org/abs/2306.04948), [code](https://github.com/wywyWang/CoachAI-Projects)).
- **TrackNetV3**: pretrained shuttle tracker, used as the Phase 1 baseline ([repo](https://github.com/qaz812345/TrackNetV3)).
- **Ultralytics YOLOv8**: pose estimation and detection (AGPL-3.0).
