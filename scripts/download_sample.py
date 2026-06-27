#!/usr/bin/env python
"""Fetch a demo soccer clip from a direct URL into data/soccer.mp4.

Real FIFA World Cup footage is copyrighted and is **not** bundled with this repo.
Use a free / royalty-free soccer clip instead (see data/README.md for sources),
grab its direct video URL, and pass it here:

    python scripts/download_sample.py --url "https://.../free-soccer-clip.mp4"

Then run:

    python scripts/track_soccer.py --source data/soccer.mp4 --out out/soccer_tracked.mp4
"""
from __future__ import annotations

import argparse
import os
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="direct URL to a free-licensed soccer .mp4")
    ap.add_argument("--out", default="data/soccer.mp4")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"[..] downloading {args.url}")
    urllib.request.urlretrieve(args.url, args.out)
    size = os.path.getsize(args.out) / 1e6
    print(f"[ok] saved {args.out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
