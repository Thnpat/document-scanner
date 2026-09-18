"""
scanner.utils
~~~~~~~~~~~~~
Low-level geometry and image helpers for the document-scanning pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Corner ordering
# ---------------------------------------------------------------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    """Return four points in consistent order: [TL, TR, BR, BL].

    Sort around the center first so symmetric/rotated quads never reuse a point.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    center = pts.mean(axis=0)
    cyclic = pts[np.argsort(np.arctan2(pts[:, 1] - center[1],
                                     pts[:, 0] - center[0]))]
    start = np.lexsort((cyclic[:, 1], cyclic.sum(axis=1)))[0]
    return np.roll(cyclic, -int(start), axis=0).copy()


# ---------------------------------------------------------------------------
# Automatic Canny thresholds
# ---------------------------------------------------------------------------

def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Apply Canny edge detection with *automatic* thresholds derived from
    the median pixel intensity.
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)


# ---------------------------------------------------------------------------
# Perspective transform
# ---------------------------------------------------------------------------

def compute_output_dimensions(rect: np.ndarray) -> tuple[int, int]:
    """Given ordered corner points, compute the width and height of the
    output rectangle that best preserves the document's aspect ratio.
    """
    (tl, tr, br, bl) = rect

    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bot))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    # Enforce A4 ratio (1 : √2 ≈ 1 : 1.4142) when the shape is roughly
    # portrait-oriented; otherwise keep the measured ratio.
    a4_ratio = 1.4142
    if max_height > max_width:
        # Portrait
        max_height = int(max_width * a4_ratio)
    else:
        # Landscape
        max_width = int(max_height * a4_ratio)

    return max(max_width, 1), max(max_height, 1)


def four_point_transform(
    image: np.ndarray,
    pts: np.ndarray,
    use_a4_ratio: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp *image* so the quadrilateral defined by *pts* maps to a
    rectangle.  Returns ``(warped, homography_matrix)``.
    """
    rect = order_points(pts)

    if use_a4_ratio:
        max_width, max_height = compute_output_dimensions(rect)
    else:
        (tl, tr, br, bl) = rect
        max_width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        max_height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        max_width = max(max_width, 1)
        max_height = max(max_height, 1)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype=np.float32)

    M, _mask = cv2.findHomography(rect, dst, cv2.RANSAC, 5.0)
    if M is None:
        # Fallback to getPerspectiveTransform (no RANSAC, but always works for
        # exactly 4 point pairs).
        M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped, M


# ---------------------------------------------------------------------------
# Resize helper
# ---------------------------------------------------------------------------

def resize_with_aspect(
    image: np.ndarray,
    max_dim: int = 1024,
) -> tuple[np.ndarray, float]:
    """Resize *image* so its longest side equals *max_dim* while keeping the
    aspect ratio.  Returns ``(resized, scale_factor)``.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image, 1.0

    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_corners(
    image: np.ndarray,
    corners: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    radius: int = 12,
    thickness: int = 3,
) -> np.ndarray:
    """Draw circles and connecting lines on a copy of *image* at *corners*."""
    vis = image.copy()
    pts = order_points(corners).astype(int)

    # Draw filled semi-transparent overlay for the detected quad
    overlay = vis.copy()
    cv2.fillConvexPoly(overlay, pts, (*color, 40))
    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

    # Draw edges
    for i in range(4):
        cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), color, thickness, cv2.LINE_AA)

    # Draw corner dots
    labels = ["TL", "TR", "BR", "BL"]
    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(vis, (x, y), radius, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            vis, labels[i], (x + radius + 4, y - radius + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

    return vis
