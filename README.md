<h1 align="center">⚽ DinoV2_Meta_VLM</h1>

<p align="center">
  <b>RF-DETR (DINOv2 backbone) + ByteTrack — real-time soccer multi-object tracking</b><br>
  <i>plus the original DINOv2-as-backbone classification study.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Detector-RF--DETR-7B61FF?logo=roboflow&logoColor=white">
  <img src="https://img.shields.io/badge/Backbone-DINOv2-1877F2?logo=meta&logoColor=white">
  <img src="https://img.shields.io/badge/Tracking-ByteTrack-00B8D9">
  <img src="https://img.shields.io/badge/Supervision-Roboflow-A259FF">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <a href="https://colab.research.google.com/github/AliSheheryar/DinoV2_Meta_VLM/blob/main/notebooks/RFDETR_DINOv2_Soccer_Tracking.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

---

## 🎯 What's inside

| Module | What it does |
|---|---|
| ⚽ **RF-DETR + ByteTrack tracking** | Detect **players** & the **ball** with RF-DETR (a **DINOv2**-backbone DETR), then track identities across frames with ByteTrack — ellipses, IDs, motion traces, ball marker |
| 🧬 **DINOv2 classification study** | The original notebook benchmarking a **DINOv2** self-supervised backbone vs. CNN classifiers on accuracy & efficiency |

> **Why DINOv2 here?** RF-DETR's image encoder is a **DINOv2** (DINOv2-with-registers) backbone —
> the same self-supervised foundation model studied in this repo, now powering a real-time
> detection + tracking pipeline instead of plain classification.

---

## 🧩 How the tracking works

```
 video frame ─► RF-DETR (DINOv2 backbone) ─► detections (players + ball)
                                                   │
                                       ByteTrack update_with_detections
                                                   │
                          Ellipse + Trace + Label + Triangle annotators
                                                   │
                                          annotated frame ─► out/*.mp4
```

- `rfdetr_tracking/tracker.py` — `SoccerTracker`: RF-DETR inference, class filtering, ByteTrack, rendering
- `scripts/track_soccer.py` — CLI: clip → annotated mp4 (+ optional GIF)
- `scripts/download_sample.py` — fetch a free-licensed clip by URL
- `notebooks/RFDETR_DINOv2_Soccer_Tracking.ipynb` — one-click **Colab** demo
- `VLM_DinoV2(1).ipynb` — the original DINOv2 classification benchmark

---

## 🚀 Quickstart

### ▶️ Colab (easiest — GPU, zero setup)
Click the **Open in Colab** badge above and run top to bottom.

### 💻 Local
```bash
pip install -r requirements.txt

# get a FREE soccer clip (real FIFA footage is copyrighted — see data/README.md)
python scripts/download_sample.py --url "https://.../free-soccer-clip.mp4"
# ...or just drop your own clip at data/soccer.mp4

# track players + ball, save annotated video (+ a README GIF)
python scripts/track_soccer.py --source data/soccer.mp4 \
    --out out/soccer_tracked.mp4 --gif assets/demo_soccer.gif
```

`--model large` for higher accuracy, `--device cuda` on a GPU, `--conf` to tune the threshold.

---

## ⚖️ Footage & licensing

Real **FIFA World Cup broadcast footage is copyrighted** and is **not** committed here.
The demo uses **free / royalty-free** soccer clips (Pexels, Pixabay, Mixkit, Roboflow Sports —
see [`data/README.md`](data/README.md)). Point the tracker at your own match clip locally if you
hold the rights.

---

## 🔭 Extending it
RF-DETR is COCO-pretrained, so it tracks `person` + `sports ball` out of the box. For
**jersey numbers, team assignment, or pitch keypoints**, fine-tune RF-DETR on a soccer dataset —
the [Roboflow Sports](https://github.com/roboflow/sports) repo has annotated data and recipes.
