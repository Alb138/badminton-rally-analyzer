"""Local Gradio app: upload a rally clip, click the 4 court corners, get the annotated video and stats.

Run: uv run python -m badminton_analyzer.app
"""
import functools
import json
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from badminton_analyzer.court import calibrate
from badminton_analyzer.pipeline import build_default_components, load_pipeline_config, run_pipeline
from badminton_analyzer.video_io import iter_frames, read_video_meta

DEFAULT_CONFIG = "configs/pipeline.yaml"
MARKER_COLOR = (255, 40, 40)  # RGB


def add_corner(points: list[tuple[int, int]], xy: tuple[int, int]) -> list[tuple[int, int]]:
    points = [tuple(p) for p in points]
    return points if len(points) >= 4 else [*points, (int(xy[0]), int(xy[1]))]


def draw_corner_markers(frame_rgb: np.ndarray, points) -> np.ndarray:
    out = frame_rgb.copy()
    radius = max(4, out.shape[1] // 150)
    for i, (x, y) in enumerate(points):
        cv2.circle(out, (int(x), int(y)), radius, MARKER_COLOR, -1)
        cv2.putText(out, str(i + 1), (int(x) + radius + 2, int(y) - radius), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.4, out.shape[1] / 1600), MARKER_COLOR, 2, cv2.LINE_AA)
    return out


def frame_at(video_path: str, index: int) -> np.ndarray:
    """RGB frame at ``index`` (or the last frame if the clip is shorter).

    Broadcast clips often open on a replay camera, so the user picks the frame to calibrate on.
    """
    frame = None
    for i, frame in enumerate(iter_frames(Path(video_path))):
        if i >= index:
            break
    if frame is None:
        raise ValueError(f"could not read a frame from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@functools.lru_cache(maxsize=2)
def get_components(config_path: str):
    return build_default_components(load_pipeline_config(Path(config_path)))


def analyze(
    video_path: str, points, config_path: str, output_root: Path = Path("outputs/app")
) -> tuple[str | None, dict | None, str]:
    if not video_path:
        return None, None, "Upload a rally clip first."
    if len(points) < 4:
        return None, None, f"Click all 4 court corners first ({len(points)}/4)."
    try:
        cfg = load_pipeline_config(Path(config_path))
        tracker, pose, classifier = get_components(str(config_path))
        out_dir = Path(output_root) / f"{datetime.now():%Y%m%d-%H%M%S}-{Path(video_path).stem}"
        outputs = run_pipeline(Path(video_path), calibrate(points), out_dir, tracker, pose, classifier, cfg)
        stats = json.loads(outputs.stats_path.read_text(encoding="utf-8"))
    except Exception as exc:  # shown to the user instead of crashing the app
        return None, None, f"Error: {exc}"
    return str(outputs.video_path), stats, f"Done: {stats['n_shots']} shots detected in {stats['duration_s']} s of video."


def build_app(config_path: str = DEFAULT_CONFIG) -> gr.Blocks:
    with gr.Blocks(title="Badminton Rally Analyzer") as demo:
        gr.Markdown("# Badminton Rally Analyzer\nUpload a singles rally clip, click the 4 outer court corners, then Analyze.")
        points = gr.State([])
        base_frame = gr.State(None)
        with gr.Row():
            video_in = gr.Video(label="Rally clip", sources=["upload"])
            with gr.Column():
                frame_img = gr.Image(label="Click the 4 outer court corners", type="numpy", interactive=False)
                frame_slider = gr.Slider(0, 1, value=0, step=1,
                                         label="Calibration frame (pick one showing the main court view)")
        with gr.Row():
            reset_btn = gr.Button("Reset corners")
            analyze_btn = gr.Button("Analyze", variant="primary")
        status = gr.Markdown()
        with gr.Row():
            video_out = gr.Video(label="Annotated rally")
            stats_out = gr.JSON(label="Rally stats")

        def on_upload(video_path):
            if not video_path:
                return None, None, [], "", gr.Slider(maximum=1, value=0)
            frame = frame_at(video_path, 0)
            last = max(read_video_meta(Path(video_path)).n_frames - 1, 1)
            return frame, frame, [], "Click the 4 outer court corners (0/4).", gr.Slider(maximum=last, value=0)

        def on_frame(video_path, index):
            if not video_path:
                return None, None, [], "Upload a rally clip first."
            frame = frame_at(video_path, int(index))
            return frame, frame, [], f"Frame {int(index)}: click the 4 outer court corners (0/4)."

        def on_click(base, pts, evt: gr.SelectData):
            if base is None:
                return None, pts, "Upload a rally clip first."
            pts = add_corner(pts, tuple(evt.index))
            return draw_corner_markers(base, pts), pts, f"Corners: {len(pts)}/4"

        video_in.change(on_upload, inputs=video_in, outputs=[frame_img, base_frame, points, status, frame_slider])
        frame_slider.release(on_frame, inputs=[video_in, frame_slider], outputs=[frame_img, base_frame, points, status])
        frame_img.select(on_click, inputs=[base_frame, points], outputs=[frame_img, points, status])
        reset_btn.click(lambda base: (base, [], "Corners reset (0/4)."), inputs=base_frame,
                        outputs=[frame_img, points, status])
        analyze_btn.click(lambda v, pts: analyze(v, pts, config_path), inputs=[video_in, points],
                          outputs=[video_out, stats_out, status])
    return demo


def main() -> None:
    build_app().launch()


if __name__ == "__main__":
    main()
