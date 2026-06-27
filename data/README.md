# Demo footage

⚠️ **Real FIFA World Cup broadcast footage is copyrighted** and is intentionally
**not** included in this repo. Use a free / royalty-free soccer clip for the public
demo, then point the tracker at your own match clip locally if you have rights to it.

## Free, license-clean soccer clips
- **Pexels** — https://www.pexels.com/search/videos/soccer/ (Pexels License, free incl. commercial)
- **Pixabay** — https://pixabay.com/videos/search/football/ (Pixabay License)
- **Mixkit** — https://mixkit.co/free-stock-video/soccer/ (Mixkit Free License)
- **Roboflow Sports** — https://github.com/roboflow/sports (sample soccer videos + datasets for exactly this task)

## Use it

```bash
# Option A — direct URL (grab a free clip's .mp4 link from a source above)
python scripts/download_sample.py --url "https://.../free-soccer-clip.mp4"

# Option B — just drop your own file here as data/soccer.mp4

# then track:
python scripts/track_soccer.py --source data/soccer.mp4 --out out/soccer_tracked.mp4 --gif assets/demo_soccer.gif
```

Tip: RF-DETR is COCO-pretrained, so it tracks **players** (`person`) and the
**`sports ball`** out of the box. For team/jersey-number or pitch-keypoint analysis,
fine-tune RF-DETR on a soccer dataset (the Roboflow Sports repo has annotated ones).
