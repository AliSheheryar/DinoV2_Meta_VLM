#!/usr/bin/env python
"""Run RF-DETR (DINOv2 backbone) + ByteTrack on a soccer clip and save annotated video.

Examples
--------
    # process a clip into an annotated mp4
    python scripts/track_soccer.py --source data/soccer.mp4 --out out/soccer_tracked.mp4

    # bigger/more-accurate model, higher threshold, GPU
    python scripts/track_soccer.py --source data/soccer.mp4 --model large --conf 0.6 --device cuda

    # also export a short GIF for the README
    python scripts/track_soccer.py --source data/soccer.mp4 --gif assets/demo_soccer.gif
"""
from __future__ import annotations

import argparse
import os
import sys

import supervision as sv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rfdetr_tracking import SoccerTracker  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="RF-DETR + ByteTrack soccer tracking")
    ap.add_argument("--source", required=True, help="path to a soccer video")
    ap.add_argument("--out", default="out/soccer_tracked.mp4")
    ap.add_argument("--model", default="base", choices=["base", "large"])
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--device", default=None, help='e.g. "cuda" (auto if omitted)')
    ap.add_argument("--gif", default=None, help="also write a GIF to this path")
    ap.add_argument("--gif-stride", type=int, default=3)
    ap.add_argument("--gif-scale", type=float, default=0.5)
    ap.add_argument("--gif-fps", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tracker = SoccerTracker(model_size=args.model, conf=args.conf, device=args.device)

    gif_frames: list = []

    def callback(frame, index):
        annotated, _ = tracker.process_frame(frame)
        if args.gif and index % args.gif_stride == 0:
            import cv2
            small = cv2.resize(annotated, None, fx=args.gif_scale, fy=args.gif_scale)
            gif_frames.append(small[:, :, ::-1])  # BGR -> RGB
        return annotated

    print(f"[..] tracking {args.source} with RF-DETR-{args.model} (DINOv2 backbone)")
    sv.process_video(source_path=args.source, target_path=args.out, callback=callback)
    print(f"[ok] wrote {args.out}")

    if args.gif and gif_frames:
        import imageio
        os.makedirs(os.path.dirname(args.gif) or ".", exist_ok=True)
        imageio.mimsave(args.gif, gif_frames, fps=args.gif_fps, loop=0)
        print(f"[ok] wrote {args.gif} ({len(gif_frames)} frames)")


if __name__ == "__main__":
    main()
