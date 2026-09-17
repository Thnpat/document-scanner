"""
scanner
~~~~~~~
Document Scanner & Perspective Rectifier package.

Provides an end-to-end computer-vision pipeline that detects tilted paper
corners in a photograph and rectifies the document to a flat top-down A4
perspective.
"""

from scanner.pipeline import (
    DetectionResult,
    PipelineResult,
    detect_document,
    enhance_scan,
    rectify_document,
    run_full_pipeline,
)
from scanner.utils import (
    auto_canny,
    draw_corners,
    four_point_transform,
    order_points,
    resize_with_aspect,
)

__all__ = [
    "DetectionResult",
    "PipelineResult",
    "auto_canny",
    "detect_document",
    "draw_corners",
    "enhance_scan",
    "four_point_transform",
    "order_points",
    "rectify_document",
    "resize_with_aspect",
    "run_full_pipeline",
]
