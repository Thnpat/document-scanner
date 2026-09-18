"""
scanner.pipeline
~~~~~~~~~~~~~~~~
End-to-end document detection and perspective-rectification pipeline.

The existing detection paths are:

1. **Contour-based** (primary) — fast, works well when the document has clear
   edges against a contrasting background.
2. **Full frame** (fallback) — accepts a genuinely cropped page.
3. **ORB feature-matching** (fallback) — uses ORB keypoints + BFMatcher with
   ratio test against a synthetic A4 template to estimate a homography when
   contour detection fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
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

logger = logging.getLogger(__name__)


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

def _preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare the original and locally normalized grayscale for edge passes."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = cv2.bilateralFilter(blurred, 9, 75, 75)
    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    normalized = cv2.GaussianBlur(normalized, (5, 5), 0)
    return gray, blurred, normalized


def _validate_quad(points: np.ndarray, shape: tuple[int, ...],
                   min_area_ratio: float = 0.12) -> tuple[np.ndarray | None, str]:
    """Reject clipped, degenerate, implausibly small or thin page candidates."""
    h, w = shape[:2]
    if points is None or np.shape(points) != (4, 2) or not np.isfinite(points).all():
        return None, "not four finite corners"
    if not cv2.isContourConvex(np.asarray(points, np.float32).reshape(-1, 1, 2)):
        return None, "non-convex or crossing corners"
    corners = order_points(points)
    distances = np.linalg.norm(corners[:, None] - corners[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    if np.min(distances) < .025 * min(h, w):
        return None, "corners too close together"
    if not cv2.isContourConvex(corners.astype(np.float32).reshape(-1, 1, 2)):
        return None, "non-convex or crossing corners"
    area_ratio = abs(cv2.contourArea(corners)) / (h * w)
    if not min_area_ratio <= area_ratio <= .98:
        return None, f"area ratio {area_ratio:.3f} outside range"
    edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    if min(edges) < .08 * min(h, w):
        return None, "edge too short"
    if max(edges) / min(edges) > 5:
        return None, "extreme perspective compression"
    aspect = (edges[0] + edges[2]) / (edges[1] + edges[3])
    if not .25 <= aspect <= 4:
        return None, f"implausible aspect {aspect:.2f}"
    for i in range(4):
        a = corners[(i - 1) % 4] - corners[i]
        b = corners[(i + 1) % 4] - corners[i]
        sine = abs(a[0] * b[1] - a[1] * b[0]) / (np.linalg.norm(a) * np.linalg.norm(b))
        if sine < .23:
            return None, "nearly collinear corners"
    border_margin = max(2, .004 * min(h, w))
    if np.any(corners[:, 0] < border_margin) or np.any(corners[:, 0] > w - 1 - border_margin) or \
            np.any(corners[:, 1] < border_margin) or np.any(corners[:, 1] > h - 1 - border_margin):
        return None, "candidate touches image edge"
    return corners, "valid"


def _boundary_contrast(gray: np.ndarray, corners: np.ndarray) -> float:
    """Compare surfaces beyond the contour line, not the line's ink itself."""
    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
    near = max(11, int(min(h, w) * .03)) | 1
    far = max(23, int(min(h, w) * .07)) | 1
    near_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (near, near))
    far_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (far, far))
    inside = cv2.subtract(cv2.erode(mask, near_kernel),
                          cv2.erode(mask, far_kernel)) > 0
    outside = cv2.subtract(cv2.dilate(mask, far_kernel),
                           cv2.dilate(mask, near_kernel)) > 0
    if not inside.any() or not outside.any():
        return 0.0
    return abs(float(np.median(gray[inside])) - float(np.median(gray[outside])))


def _page_clipped_by_frame(image: np.ndarray) -> bool:
    """Flag a likely cut page only after no complete quadrilateral was found."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    border_level = float(np.median(border))
    if border_level >= 235:
        return False
    fractions = []
    for edge_name, (g, s) in zip(("top", "bottom", "left", "right"),
                                  ((gray[0], saturation[0]),
                                   (gray[-1], saturation[-1]),
                                   (gray[:, 0], saturation[:, 0]),
                                   (gray[:, -1], saturation[:, -1]))):
        paper_fraction = float(np.mean((g >= border_level + 18) & (s < 35)))
        fractions.append(paper_fraction)
        logger.debug("Possible clipped page edge=%s paper_fraction=%.3f", edge_name,
                     paper_fraction)
    # A single bright edge can be a lit background or a shadow; require an
    # opposing pair before treating border color as evidence of cropping.
    return ((fractions[0] >= .025 and fractions[1] >= .025) or
            (fractions[2] >= .025 and fractions[3] >= .025))


def detect_document_contour(image: np.ndarray) -> DetectionResult:
    """Primary detection: edge-based contour approach."""
    debug: dict[str, np.ndarray] = {}

    h, w = image.shape[:2]
    logger.debug("Contour detection image=%dx%d", w, h)
    logger.debug("Contour preprocessing: grayscale Gaussian=5 bilateral=9/75/75 "
                 "CLAHE=2.0/8x8 normalized_Gaussian=5")
    gray, blurred, normalized = _preprocess(image)
    debug["grayscale"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    edged_auto = auto_canny(blurred)
    debug["edges_auto"] = cv2.cvtColor(edged_auto, cv2.COLOR_GRAY2BGR)
    close3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    passes = [
        ("auto", edged_auto),
        ("morphed", cv2.morphologyEx(edged_auto, cv2.MORPH_CLOSE, close5)),
        ("tight", cv2.morphologyEx(cv2.Canny(blurred, 30, 120),
                                   cv2.MORPH_CLOSE, close3)),
        ("normalized", cv2.morphologyEx(cv2.Canny(normalized, 20, 80),
                                        cv2.MORPH_CLOSE, close3)),
        ("low_contrast", cv2.morphologyEx(cv2.Canny(normalized, 12, 45),
                                          cv2.MORPH_CLOSE, close7)),
        ("adaptive", cv2.morphologyEx(cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5), cv2.MORPH_CLOSE, close3)),
    ]
    logger.debug("Contour passes: auto_canny; auto_close=5; tight_canny=30/120 "
                 "close=3; normalized_canny=20/80 close=3; low_contrast_canny=12/45 "
                 "close=7; adaptive_block=21 C=5 close=3; epsilons=.012/.02/.03/.045")
    debug["edges_morphed"] = cv2.cvtColor(passes[1][1], cv2.COLOR_GRAY2BGR)
    debug["edges_tight"] = cv2.cvtColor(passes[2][1], cv2.COLOR_GRAY2BGR)
    debug["edges_low_contrast"] = cv2.cvtColor(passes[4][1], cv2.COLOR_GRAY2BGR)
    debug["adaptive_threshold"] = cv2.cvtColor(passes[5][1], cv2.COLOR_GRAY2BGR)
    candidates: list[tuple[float, np.ndarray, str]] = []
    for name, edged in passes:
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        logger.debug("Contour pass=%s contours=%d", name, len(contours))
        plausible = 0
        too_small = 0
        for contour_index, cnt in enumerate(sorted(contours, key=cv2.contourArea,
                                                    reverse=True)[:60]):
            area = cv2.contourArea(cnt)
            if area < .10 * h * w:
                too_small += 1
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                logger.debug("%s contour=%d rejected: zero perimeter", name, contour_index)
                continue
            for epsilon in (.012, .02, .03, .045):
                approx = cv2.approxPolyDP(cnt, epsilon * perimeter, True)
                if len(approx) != 4:
                    logger.debug("%s contour=%d area=%.0f ratio=%.3f perimeter=%.0f "
                                 "epsilon=%.3f vertices=%d rejected", name, contour_index,
                                 area, area / (h * w), perimeter, epsilon, len(approx))
                    continue
                corners, reason = _validate_quad(approx.reshape(4, 2), image.shape)
                if corners is None:
                    logger.debug("%s contour=%d area=%.0f ratio=%.3f perimeter=%.0f "
                                 "epsilon=%.3f vertices=4 convex=%s corners=%s rejected: %s",
                                 name, contour_index, area, area / (h * w), perimeter,
                                 epsilon, cv2.isContourConvex(approx),
                                 approx.reshape(4, 2).tolist(), reason)
                    continue
                ratio = cv2.contourArea(corners) / (h * w)
                fit = min(area, cv2.contourArea(corners)) / max(area, cv2.contourArea(corners))
                contrast = _boundary_contrast(gray, corners)
                if contrast < 8:
                    logger.debug("%s contour=%d ratio=%.3f rejected: weak boundary contrast %.1f",
                                 name, contour_index, ratio, contrast)
                    continue
                score = .75 * ratio + .15 * fit + .10 * min(contrast / 50, 1)
                border_distances = [float(np.min(corners[:, 0])),
                                    float(w - 1 - np.max(corners[:, 0])),
                                    float(np.min(corners[:, 1])),
                                    float(h - 1 - np.max(corners[:, 1]))]
                logger.debug("%s contour=%d area=%.0f perimeter=%.0f epsilon=%.3f "
                             "vertices=4 convex=True contour_ratio=%.3f quad_ratio=%.3f "
                             "fit=%.2f contrast=%.1f score=%.3f border_LRTB=%s corners=%s",
                             name, contour_index, area, perimeter, epsilon,
                             area / (h * w), ratio, fit, contrast, score,
                             np.round(border_distances, 1).tolist(),
                             np.rint(corners).astype(int).tolist())
                candidates.append((score, corners, name))
                plausible += 1
                break
        logger.debug("Contour pass=%s after_area_filter=%d quadrilaterals=%d below-min-area=%d",
                     name, min(len(contours), 60) - too_small, plausible, too_small)

    if candidates:
        score, corners, selected_pass = max(candidates, key=lambda c: c[0])
        logger.debug("Contour selected pass=%s score=%.3f area=%.3f candidates=%d corners=%s",
                     selected_pass, score, cv2.contourArea(corners) / (h * w),
                     len(candidates), np.rint(corners).astype(int).tolist())
        # Sub-pixel corner refinement
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01
        )
        corners_refined = cv2.cornerSubPix(
            gray, corners.copy().astype(np.float32),
            (5, 5), (-1, -1), criteria,
        )
        refined, reason = _validate_quad(corners_refined, image.shape)
        if refined is not None and np.max(np.linalg.norm(refined - corners, axis=1)) < 10:
            corners = refined
        else:
            logger.debug("Retaining unrefined contour corners: %s", reason)
        corners_overlay = draw_corners(image, corners)
        debug["corners_detected"] = corners_overlay

        return DetectionResult(
            success=True,
            corners=corners,
            method="contour",
            message="Document detected via contour analysis.",
            debug_images=debug,
        )

    clipped = _page_clipped_by_frame(image)
    logger.debug("Contour detection failed: no valid quadrilateral; bright-border heuristic=%s", clipped)
    return DetectionResult(
        success=False,
        method="contour",
        message=("The page may cross the image edge; show all four corners."
                 if clipped else "No reliable page boundary found."),
        debug_images=debug,
    )


# ---------------------------------------------------------------------------
# 2. Full-frame page fallback
# ---------------------------------------------------------------------------

def detect_full_frame_page(image: np.ndarray) -> DetectionResult:
    """Accept a cropped page only when all edges share a clean paper surface."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    size = max(8, int(min(h, w) * .04))
    patches = (
        image[:size, :size],
        image[:size, -size:],
        image[-size:, -size:],
        image[-size:, :size],
    )
    brightness = []
    for patch in patches:
        pg = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        ps = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1]
        if np.median(pg) < 230 or np.std(pg) > 18 or np.median(ps) > 35:
            logger.debug("Full frame rejected: corner brightness=%.1f std=%.1f saturation=%.1f",
                         np.median(pg), np.std(pg), np.median(ps))
            return DetectionResult(success=False, method="frame",
                                   message="The page does not fill the image frame.")
        brightness.append(float(np.median(pg)))

    if max(brightness) - min(brightness) > 25:
        logger.debug("Full frame rejected: corner brightness range %.1f", max(brightness)-min(brightness))
        return DetectionResult(success=False, method="frame",
                               message="The page edges have uneven lighting.")

    strips = ((gray[:size], sat[:size]), (gray[-size:], sat[-size:]),
              (gray[:, :size], sat[:, :size]), (gray[:, -size:], sat[:, -size:]))
    for index, (g, s) in enumerate(strips):
        paper_fraction = np.mean((g >= 225) & (s <= 45))
        if paper_fraction < .93:
            logger.debug("Full frame rejected: border %d paper fraction %.3f", index, paper_fraction)
            return DetectionResult(False, method="frame", message="Background is visible at the image edge.")

    center = gray[h//5:4*h//5, w//5:4*w//5]
    center_brightness = float(np.median(center))
    edge_brightness = float(np.median(np.concatenate(
        (gray[:size].ravel(), gray[-size:].ravel(),
         gray[:, :size].ravel(), gray[:, -size:].ravel()))))
    ink_fraction = float(np.mean(center < 160))
    if abs(center_brightness - edge_brightness) > 8 or not .002 <= ink_fraction <= .45:
        logger.debug("Full frame rejected: center=%.1f edge=%.1f ink=%.3f",
                     center_brightness, edge_brightness, ink_fraction)
        return DetectionResult(False, method="frame",
                               message="Not enough evidence that the page fills the frame.")

    logger.debug("Full frame accepted: %dx%d edge=%.1f ink=%.3f",
                 w, h, edge_brightness, ink_fraction)

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
        nfeatures=2500,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=5,
        patchSize=31,
        fastThreshold=5,
    )

    # Detect keypoints in the input image
    kp_img, des_img = orb.detectAndCompute(enhanced, None)
    logger.debug("ORB image=%dx%d keypoints=%d descriptors=%s", image.shape[1],
                 image.shape[0], len(kp_img), None if des_img is None else des_img.shape)

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
    logger.debug("ORB template keypoints=%d descriptors=%s", len(kp_tmpl),
                 None if des_tmpl is None else des_tmpl.shape)

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

    logger.debug("ORB raw pairs=%d complete_pairs=%d ratio=0.75 good_matches=%d",
                 len(raw_matches), sum(len(pair) == 2 for pair in raw_matches), len(good_matches))

    if len(good_matches) < 10:
        return DetectionResult(
            success=False,
            method="orb",
            message=f"Insufficient good matches ({len(good_matches)}). Need at least 10.",
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
    logger.debug("ORB RANSAC inliers=%d/%d", inliers, len(good_matches))
    if inliers < 8 or inliers / len(good_matches) < .55:
        return DetectionResult(False, method="orb", message="ORB geometry has too few consistent matches.",
                               debug_images=debug, orb_matches=len(good_matches))
    inlier_template = pts_tmpl[mask.ravel() != 0].reshape(-1, 2)
    if (np.ptp(inlier_template[:, 0]) < .40 * template.shape[1] or
            np.ptp(inlier_template[:, 1]) < .40 * template.shape[0]):
        return DetectionResult(False, method="orb", message="ORB matches cover too little of the page.",
                               debug_images=debug, orb_matches=len(good_matches))

    # Map template corners through the homography → document corners
    th, tw = template.shape[:2]
    template_corners = np.float32([
        [0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1],
    ]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(template_corners, M).reshape(4, 2)

    # Validate the mapped quadrilateral
    mapped, reason = _validate_quad(mapped, image.shape)
    if mapped is None:
        logger.debug("ORB rejected geometry: %s", reason)
        return DetectionResult(False, method="orb", message=f"ORB geometry rejected: {reason}.",
                               debug_images=debug, orb_matches=len(good_matches))

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
        logger.debug("Detection selected method=contour corners=%s",
                     np.rint(result.corners).astype(int).tolist())
        return result
    logger.debug("Contour fallback triggered: %s", result.message)

    # Merge debug images from the contour attempt
    contour_debug = result.debug_images.copy()

    frame_result = detect_full_frame_page(image)
    if frame_result.success:
        frame_result.debug_images = contour_debug
        logger.debug("Detection selected method=frame corners=%s",
                     frame_result.corners.astype(int).tolist())
        return frame_result
    logger.debug("Full frame fallback rejected: %s; trying ORB", frame_result.message)

    orb_result = detect_document_orb(image)
    orb_result.debug_images = {**contour_debug, **orb_result.debug_images}

    if orb_result.success:
        orb_result.message = (
            "Contour detection failed — fell back to ORB feature matching. "
            + orb_result.message
        )
        logger.debug("Detection selected method=orb corners=%s",
                     np.rint(orb_result.corners).astype(int).tolist())
        return orb_result

    logger.debug("Detection failed: contour=%s frame=%s orb=%s",
                 result.message, frame_result.message, orb_result.message)
    if not orb_result.success:
        orb_result.message = ("Could not reliably detect all four page corners. "
                              "Make sure the entire page is visible."
                              if "image edge" in result.message else
                              "Could not reliably detect the document boundary.")
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
    logger.debug("Pipeline image=%dx%d working=%dx%d scale=%.4f enhancement=%s A4=%s",
                 image.shape[1], image.shape[0], working.shape[1], working.shape[0],
                 scale, enhance_mode, use_a4_ratio)

    detection = detect_document(working)

    if not detection.success:
        logger.debug("Pipeline detection failed: %s", detection.message)
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
    logger.debug("Pipeline rectified output=%dx%d method=%s",
                 warped.shape[1], warped.shape[0], detection.method)

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
