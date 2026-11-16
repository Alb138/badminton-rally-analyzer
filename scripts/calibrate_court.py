"""Click the 4 outer court corners on a clip frame and save the calibration.

Usage: uv run python scripts/calibrate_court.py VIDEO --out CAL.json [--frame N]

Broadcast clips often open on a replay camera; use --frame to pick a frame showing the main view.

Left-click the outer (doubles) corners in any order. Keys: r = reset,
Enter = save (needs 4 points), Esc = quit without saving.
"""
import argparse
import sys
from itertools import islice
from pathlib import Path

import cv2

from badminton_analyzer.court import calibrate, save_calibration
from badminton_analyzer.video_io import iter_frames

WINDOW = "calibrate court: click 4 outer corners, Enter=save, r=reset, Esc=quit"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0, help="frame index to calibrate on (default 0)")
    args = parser.parse_args()

    frame = next(islice(iter_frames(args.video), args.frame, None), None)
    if frame is None:
        print(f"could not read frame {args.frame} from {args.video}", file=sys.stderr)
        return 1
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        view = frame.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(view, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(view, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            print("quit without saving")
            return 1
        if key == ord("r"):
            points.clear()
        if key in (10, 13) and len(points) == 4:
            try:
                cal = calibrate(points)
            except ValueError as exc:
                print(exc, file=sys.stderr)
                points.clear()
                continue
            save_calibration(cal, args.out)
            cv2.destroyAllWindows()
            print(f"saved {args.out}")
            return 0


if __name__ == "__main__":
    sys.exit(main())
