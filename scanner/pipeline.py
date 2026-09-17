"""
scanner.pipeline
~~~~~~~~~~~~~~~~
End-to-end document detection and perspective-rectification pipeline.

Two detection strategies are provided:

1. **Contour-based** (primary) — fast, works well when the document has clear
   edges against a contrasting background.
2. **ORB feature-matching** (fallback) — uses ORB keypoints + BFMatcher with
   ratio test against a synthetic A4 template to estimate a homography when
   contour detection fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from scanner.utils import (
    auto_canny,
    compute_output_dimensions,
    draw_corners,
    four_point_transform,
    order_points,
    resize_with_aspect,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Outcome of the document-detection stage."""
    success: bool
    corners: np.ndarray | None = None
    method: str = ""
    message: str = ""
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)
    orb_matches: int = 0


@dataclass
class PipelineResult:
    """Outcome of the full scan pipeline."""
    success: bool
    original: np.ndarray | None = None
    corners_overlay: np.ndarray | None = None
    warped: np.ndarray | None = None
    enhanced: np.ndarray | None = None
    homography: np.ndarray | None = None
    detection: DetectionResult | None = None
    message: str = ""


# ---------------------------------------------------------------------------
# 1. Contour-based detection
# ---------------------------------------------------------------------------

def _preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert to grayscale, blur, and return (gray, blurred)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = cv2.bilateralFilter(blurred, 9, 75, 75)
    return gray, blurred


def _find_document_contour(
    edged: np.ndarray,
    image_area: int,
    min_area_ratio: float = 0.05,
) -> np.ndarray | None:
    """Find the largest 4-sided contour whose area exceeds *min_area_ratio*
    of the total image area.
    """
    contours, _ = cv2.findContours(
        edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < image_area * min_area_ratio:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None


def detect_document_contour(image: np.ndarray) -> DetectionResult:
    """Primary detection: edge-based contour approach."""
    debug: dict[str, np.ndarray] = {}

    gray, blurred = _preprocess(image)
    debug["grayscale"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # --- Multi-pass edge detection for robustness ---
    edged_auto = auto_canny(blurred)
    debug["edges_auto"] = cv2.cvtColor(edged_auto, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    image_area = h * w

    # Try auto-Canny first
    corners = _find_document_contour(edged_auto, image_area)

    if corners is None:
        # Try with morphological closing to bridge gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edged_auto, cv2.MORPH_CLOSE, kernel, iterations=2)
        debug["edges_morphed"] = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
        corners = _find_document_contour(closed, image_area)

    if corners is None:
        # Try tighter Canny thresholds
        edged_tight = cv2.Canny(blurred, 30, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edged_tight = cv2.morphologyEx(edged_tight, cv2.MORPH_CLOSE, kernel, iterations=2)
        debug["edges_tight"] = cv2.cvtColor(edged_tight, cv2.COLOR_GRAY2BGR)
        corners = _find_document_contour(edged_tight, image_area)

    if corners is None:
        # Try adaptive threshold approach
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=3)
        debug["adaptive_threshold"] = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
        corners = _find_document_contour(adaptive, image_area)

    if corners is not None:
        # Sub-pixel corner refinement
        gray_for_refine = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01
        )
        corners_refined = cv2.cornerSubPix(
            gray_for_refine,
            corners.astype(np.float32),
            (5, 5), (-1, -1), criteria,
        )
        corners_overlay = draw_corners(image, corners_refined)
        debug["corners_detected"] = corners_overlay

        return DetectionResult(
            success=True,
            corners=corners_refined,
            method="contour",
            message="Document detected via contour analysis.",
            debug_images=debug,
        )

    return DetectionResult(
        success=False,
        method="contour",
        message="No rectangular contour found.",
        debug_images=debug,
    )


# ---------------------------------------------------------------------------
# 2. Full-frame page fallback
# ---------------------------------------------------------------------------

def detect_full_frame_page(image: np.ndarray) -> DetectionResult:
    """Accept an already cropped scan when paper reaches all four corners.

    With no visible outer edge, contour detection cannot locate the page. Check
    the corner regions for a bright, fairly uniform, low-saturation surface
    before treating the image bounds as the document bounds.
    """
    h, w = image.shape[:2]
    size = max(8, int(min(h, w) * 0.06))
    patches = (
        image[:size, :size],
        image[:size, -size:],
        image[-size:, -size:],
        image[-size:, :size],
    )
    brightness = []
    for patch in patches:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        saturation = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1]
        if (np.median(gray) < 110 or np.std(gray) > 30
                or np.median(saturation) > 80):
            return DetectionResult(success=False, method="frame",
                                   message="The page does not fill the image frame.")
        brightness.append(float(np.median(gray)))

    if max(brightness) - min(brightness) > 70:
        return DetectionResult(success=False, method="frame",
                               message="The page edges have uneven lighting.")

    return DetectionResult(
        success=True,
        corners=np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]),
        method="frame",
        message="The document already fills the frame; its image edges were used.",
    )


# ---------------------------------------------------------------------------
# 3. ORB feature-matching fallback
# ---------------------------------------------------------------------------

def _create_a4_template(width: int = 595, height: int = 842) -> np.ndarray:
    """Create a synthetic white A4 template with corner markers and edge
    features for ORB matching.
    """
    template = np.ones((height, width), dtype=np.uint8) * 255

    # Draw a border and corner markers to give ORB something to match
    cv2.rectangle(template, (10, 10), (width - 10, height - 10), 0, 3)

    # Corner L-shapes
    corner_len = 60
    for (cx, cy) in [(10, 10), (width - 10, 10), (width - 10, height - 10), (10, height - 10)]:
        dx = corner_len if cx < width // 2 else -corner_len
        dy = corner_len if cy < height // 2 else -corner_len
        cv2.line(template, (cx, cy), (cx + dx, cy), 0, 5)
        cv2.line(template, (cx, cy), (cx, cy + dy), 0, 5)

    return template


def detect_document_orb(image: np.ndarray) -> DetectionResult:
    """Fallback detection: ORB feature-matching against an A4 template."""
    debug: dict[str, np.ndarray] = {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast for better feature detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Create ORB detector
    orb = cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31,
    )

    # Detect keypoints in the input image
    kp_img, des_img = orb.detectAndCompute(enhanced, None)

    if des_img is None or len(kp_img) < 10:
        return DetectionResult(
            success=False,
            method="orb",
            message="Too few ORB keypoints detected in the image.",
            debug_images=debug,
        )

    # Draw keypoints for visualization
    kp_vis = cv2.drawKeypoints(
        image, kp_img, None,
        color=(0, 200, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    debug["orb_keypoints"] = kp_vis

    # Detect on the template
    template = _create_a4_template()
    kp_tmpl, des_tmpl = orb.detectAndCompute(template, None)

    if des_tmpl is None or len(kp_tmpl) < 4:
        return DetectionResult(
            success=False,
            method="orb",
            message="Template feature extraction failed.",
            debug_images=debug,
        )

    # Match with BFMatcher + ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des_tmpl, des_img, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    debug_info = f"ORB matches after ratio test: {len(good_matches)}"

    if len(good_matches) < 8:
        return DetectionResult(
            success=False,
            method="orb",
            message=f"Insufficient good matches ({len(good_matches)}). Need at least 8.",
            debug_images=debug,
            orb_matches=len(good_matches),
        )

    # Compute homography
    pts_tmpl = np.float32([kp_tmpl[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_img = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(pts_tmpl, pts_img, cv2.RANSAC, 5.0)

    if M is None:
        return DetectionResult(
            success=False,
            method="orb",
            message="Homography estimation failed (RANSAC).",
            debug_images=debug,
            orb_matches=len(good_matches),
        )

    inliers = int(mask.sum()) if mask is not None else 0

    # Map template corners through the homography → document corners
    th, tw = template.shape[:2]
    template_corners = np.float32([
        [0, 0], [tw, 0], [tw, th], [0, th],
    ]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(template_corners, M).reshape(4, 2)

    # Validate the mapped quadrilateral
    h, w = image.shape[:2]
    for pt in mapped:
        if pt[0] < -w * 0.1 or pt[0] > w * 1.1 or pt[1] < -h * 0.1 or pt[1] > h * 1.1:
            return DetectionResult(
                success=False,
                method="orb",
                message="Detected region extends far outside the image bounds.",
                debug_images=debug,
                orb_matches=len(good_matches),
            )

    corners_overlay = draw_corners(image, mapped)
    debug["orb_corners_detected"] = corners_overlay

    return DetectionResult(
        success=True,
        corners=mapped,
        method="orb",
        message=f"Document detected via ORB matching ({inliers} inliers from {len(good_matches)} matches).",
        debug_images=debug,
        orb_matches=len(good_matches),
    )


# ---------------------------------------------------------------------------
# Detection dispatcher
# ---------------------------------------------------------------------------

def detect_document(image: np.ndarray) -> DetectionResult:
    """Try contours, then a cropped page, then ORB feature matching."""
    result = detect_document_contour(image)
    if result.success:
        return result

    # Merge debug images from the contour attempt
    contour_debug = result.debug_images.copy()

    frame_result = detect_full_frame_page(image)
    if frame_result.success:
        frame_result.debug_images = contour_debug
        return frame_result

    orb_result = detect_document_orb(image)
    orb_result.debug_images = {**contour_debug, **orb_result.debug_images}

    if orb_result.success:
        orb_result.message = (
            "Contour detection failed — fell back to ORB feature matching. "
            + orb_result.message
        )

    return orb_result


# ---------------------------------------------------------------------------
# Rectification
# ---------------------------------------------------------------------------

def rectify_document(
    image: np.ndarray,
    corners: np.ndarray,
    use_a4_ratio: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp *image* to a top-down rectangle using the detected *corners*.

    Returns ``(warped_image, homography_matrix)``.
    """
    return four_point_transform(image, corners, use_a4_ratio=use_a4_ratio)


# ---------------------------------------------------------------------------
# Post-processing / enhancement
# ---------------------------------------------------------------------------

def enhance_scan(
    image: np.ndarray,
    mode: Literal["original", "grayscale", "bw_scan", "sharpen"] = "original",
    bw_block_size: int = 11,
    bw_c: int = 2,
) -> np.ndarray:
    """Apply post-processing to the rectified image.

    Modes
    -----
    - ``"original"``  — no change
    - ``"grayscale"`` — convert to grayscale
    - ``"bw_scan"``   — adaptive-threshold B&W (classic scanner look)
    - ``"sharpen"``   — apply an unsharp-mask sharpening filter
    """
    if mode == "original":
        return image.copy()

    if mode == "grayscale":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if mode == "bw_scan":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bw_block_size,
            bw_c,
        )
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    if mode == "sharpen":
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return sharpened

    return image.copy()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    image: np.ndarray,
    *,
    enhance_mode: str = "original",
    use_a4_ratio: bool = True,
    max_processing_dim: int = 1500,
) -> PipelineResult:
    """Run the complete scan pipeline: detect → rectify → enhance.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.
    enhance_mode : str
        Post-processing mode (see :func:`enhance_scan`).
    use_a4_ratio : bool
        If True, force the output to A4 aspect ratio.
    max_processing_dim : int
        Resize the image for processing if it exceeds this dimension
        (corners are scaled back to original resolution).
    """
    original = image.copy()

    # Optionally resize for faster processing
    working, scale = resize_with_aspect(image, max_processing_dim)

    detection = detect_document(working)

    if not detection.success:
        return PipelineResult(
            success=False,
            original=original,
            detection=detection,
            message=detection.message,
        )

    # Scale corners back to original resolution
    corners = detection.corners / scale

    # Draw corners on the *original* image
    corners_overlay = draw_corners(original, corners)

    # Rectify from the full-resolution original
    warped, homography = rectify_document(original, corners, use_a4_ratio=use_a4_ratio)

    # Enhance
    enhanced = enhance_scan(warped, mode=enhance_mode)

    return PipelineResult(
        success=True,
        original=original,
        corners_overlay=corners_overlay,
        warped=warped,
        enhanced=enhanced,
        homography=homography,
        detection=detection,
        message=detection.message,
    )
