"""Run TrackNetV3's predict.py with single-process data loading.

predict.py sets DataLoader(num_workers=batch_size). On Windows, worker
processes are spawned and the video dataset cannot be pickled across to them,
so inference crashes. This launcher forces num_workers=0 and then runs the
script unchanged, leaving the third-party code untouched.

Usage (cwd = TrackNetV3 repo): python tracknetv3_launcher.py predict.py [predict.py args...]
"""
import os
import runpy
import sys

from torch.utils.data import DataLoader


def force_single_process_loading() -> None:
    original_init = DataLoader.__init__

    def init(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs.pop("prefetch_factor", None)
        kwargs.pop("persistent_workers", None)
        original_init(self, *args, **kwargs)

    DataLoader.__init__ = init


def main() -> None:
    script, *script_args = sys.argv[1:]
    sys.path.insert(0, os.getcwd())
    sys.argv = [script, *script_args]
    force_single_process_loading()
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
