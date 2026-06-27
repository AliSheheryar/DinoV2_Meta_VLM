"""RF-DETR (DINOv2 backbone) + ByteTrack multi-object tracking."""
from .tracker import SoccerTracker

__all__ = ["SoccerTracker"]
__version__ = "0.1.0"
