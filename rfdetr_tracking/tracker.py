"""RF-DETR (DINOv2 backbone) detection + ByteTrack tracking.

RF-DETR is Roboflow's real-time DETR; its image encoder is a **DINOv2** backbone
(DINOv2-with-registers windowed attention). We run it per frame, then maintain
identities across frames with ByteTrack and render player ellipses, ID labels,
motion traces and a ball marker — the classic soccer-tracking visualization.

The model is decoupled from the loop, so swapping RF-DETR Base ↔ Large, or
pointing at any video (a free soccer clip, or your own World Cup match), needs no
changes to the tracking/annotation code.
"""
from __future__ import annotations

import numpy as np
import supervision as sv
from PIL import Image


class SoccerTracker:
    def __init__(
        self,
        model_size: str = "base",          # "base" or "large"
        conf: float = 0.5,
        keep_classes: tuple[str, ...] = ("person", "sports ball"),
        device: str | None = None,          # e.g. "cuda"; RF-DETR auto-selects if None
        trace_length: int = 60,
    ):
        # Imported lazily so the module is importable without the heavy deps installed.
        from rfdetr import RFDETRBase, RFDETRLarge
        from rfdetr.util.coco_classes import COCO_CLASSES

        self.model = (RFDETRLarge if model_size.lower() == "large" else RFDETRBase)()
        if device:
            try:
                self.model.model.to(device)
            except Exception:
                pass  # RF-DETR manages its own device placement in most builds

        self.conf = conf
        self.coco = COCO_CLASSES                       # {id: name}
        name2id = {v: k for k, v in COCO_CLASSES.items()}
        self.keep_ids = [name2id[c] for c in keep_classes if c in name2id]
        self.ball_id = name2id.get("sports ball")

        self.tracker = sv.ByteTrack()
        self.ellipse = sv.EllipseAnnotator(thickness=2)
        self.label = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self.trace = sv.TraceAnnotator(trace_length=trace_length, thickness=2)
        self.triangle = sv.TriangleAnnotator(base=20, height=18)  # marks the ball

    def reset(self):
        """Clear tracker state — call between separate clips."""
        self.tracker.reset()

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections:
        rgb = Image.fromarray(frame_bgr[:, :, ::-1])              # BGR -> RGB (PIL)
        detections = self.model.predict(rgb, threshold=self.conf)
        if self.keep_ids:
            detections = detections[np.isin(detections.class_id, self.keep_ids)]
        return detections

    def annotate(self, frame_bgr: np.ndarray, detections: sv.Detections) -> np.ndarray:
        out = frame_bgr.copy()
        is_ball = detections.class_id == self.ball_id
        players = detections[~is_ball]
        ball = detections[is_ball]

        out = self.trace.annotate(out, players)
        out = self.ellipse.annotate(out, players)
        if players.tracker_id is not None and len(players):
            labels = [
                f"#{tid} {self.coco.get(int(cid), int(cid))} {conf:.2f}"
                for tid, cid, conf in zip(
                    players.tracker_id, players.class_id, players.confidence
                )
            ]
            out = self.label.annotate(out, players, labels)
        if len(ball):
            out = self.triangle.annotate(out, ball)
        return out

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, sv.Detections]:
        """Detect → track → annotate one BGR frame. Returns (annotated_bgr, detections)."""
        detections = self.detect(frame_bgr)
        detections = self.tracker.update_with_detections(detections)
        return self.annotate(frame_bgr, detections), detections
