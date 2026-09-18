"""Geometric regression cases for the existing document detection methods."""

import unittest

import cv2
import numpy as np

from scanner.pipeline import (detect_document, detect_document_orb,
                              detect_full_frame_page, run_full_pipeline)
from scanner.utils import order_points


def desk(height: int, width: int, *, wood: bool = False, light: bool = False) -> np.ndarray:
    if not wood:
        return np.full((height, width, 3), 218 if light else 175, np.uint8)
    rng = np.random.default_rng(7)
    row = 168 + 8 * np.sin(np.arange(height) / 6) + rng.normal(0, 3, height)
    surface = np.broadcast_to(row[:, None], (height, width)).copy()
    surface += rng.normal(0, 2, surface.shape)
    return np.stack((surface * .82, surface * .94, surface), axis=2).clip(0, 255).astype(np.uint8)


def document(image: np.ndarray, corners: np.ndarray, *, table: bool = False,
             complex_page: bool = False, shadow: bool = False) -> np.ndarray:
    image = image.copy()
    corners = np.asarray(corners, np.float32)
    if shadow:
        shifted = corners.astype(np.int32) + (12, 16)
        cv2.fillConvexPoly(image, shifted, (188, 188, 188))
    cv2.fillConvexPoly(image, corners.astype(np.int32), (246, 246, 246))
    matrix = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), corners)

    def point(x: float, y: float) -> tuple[int, int]:
        p = cv2.perspectiveTransform(np.float32([[[x, y]]]), matrix)[0, 0]
        return tuple(np.rint(p).astype(int))

    for y in np.linspace(.13, .85, 13):
        if table and .25 < y < .75:
            continue
        cv2.line(image, point(.13, y), point(.78 if complex_page else .7, y),
                 (65, 65, 65), 2)
    if table:
        box = np.int32([point(.08, .23), point(.92, .23),
                         point(.92, .83), point(.08, .83)])
        cv2.polylines(image, [box], True, (25, 25, 25), 4)
        for y in (.39, .55, .7):
            cv2.line(image, point(.08, y), point(.92, y), (30, 30, 30), 3)
        for x in (.59, .76):
            cv2.line(image, point(x, .23), point(x, .83), (30, 30, 30), 3)
    if complex_page:
        cv2.circle(image, point(.16, .08), 12, (35, 35, 35), -1)
        cv2.rectangle(image, point(.12, .88), point(.53, .93), (60, 60, 60), 2)
    return image


class DetectionReliabilityTests(unittest.TestCase):
    def assert_page(self, image: np.ndarray, expected: np.ndarray,
                    tolerance: float = 24) -> None:
        result = run_full_pipeline(image, use_a4_ratio=False)
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.detection.method, "contour")
        scale = min(1.0, 1500 / max(image.shape[:2]))
        actual = order_points(result.detection.corners / scale)
        expected = order_points(np.asarray(expected, np.float32))
        self.assertLess(float(np.max(np.linalg.norm(actual - expected, axis=1))), tolerance)
        self.assertGreater(min(result.warped.shape[:2]), 100)
        projected = cv2.perspectiveTransform(actual[None], result.homography)[0]
        self.assertLess(float(np.max(np.linalg.norm(projected - np.float32([
            [0, 0], [result.warped.shape[1] - 1, 0],
            [result.warped.shape[1] - 1, result.warped.shape[0] - 1],
            [0, result.warped.shape[0] - 1]]), axis=1))), 2)

    def test_angled_paper_on_wood(self) -> None:
        q = np.float32([[210, 55], [680, 150], [590, 540], [115, 435]])
        self.assert_page(document(desk(600, 800, wood=True), q), q)

    def test_slight_blur_on_wood(self) -> None:
        q = np.float32([[205, 65], [670, 140], [585, 535], [120, 435]])
        image = document(desk(600, 800, wood=True), q, table=True)
        self.assert_page(cv2.GaussianBlur(image, (5, 5), 1.1), q)

    def test_invoice_table_does_not_win(self) -> None:
        q = np.float32([[160, 70], [680, 95], [650, 545], [125, 510]])
        self.assert_page(document(desk(600, 800, light=True), q, table=True), q)

    def test_light_background_with_shadow(self) -> None:
        q = np.float32([[155, 80], [660, 110], [630, 540], [125, 500]])
        self.assert_page(document(desk(600, 800, light=True), q, shadow=True), q)

    def test_dark_background(self) -> None:
        q = np.float32([[185, 65], [650, 100], [635, 535], [145, 495]])
        image = np.full((600, 800, 3), (42, 50, 58), np.uint8)
        self.assert_page(document(image, q), q)

    def test_page_around_thirty_percent(self) -> None:
        q = np.float32([[270, 135], [790, 165], [760, 650], [245, 620]])
        self.assert_page(document(desk(800, 1050, light=True), q), q)

    def test_page_around_fifty_percent(self) -> None:
        q = np.float32([[175, 90], [835, 120], [805, 645], [145, 615]])
        self.assert_page(document(desk(750, 1000, wood=True), q), q)

    def test_small_page_on_desk(self) -> None:
        q = np.float32([[290, 110], [770, 170], [710, 570], [250, 520]])
        self.assert_page(document(desk(700, 1000, wood=True), q), q)

    def test_near_full_page_still_uses_its_edges(self) -> None:
        q = np.float32([[24, 22], [770, 30], [760, 570], [30, 565]])
        self.assert_page(document(desk(600, 800), q), q)

    def test_page_around_ninety_percent(self) -> None:
        q = np.float32([[18, 16], [782, 22], [778, 584], [20, 578]])
        self.assert_page(document(desk(600, 800, light=True), q), q)

    def test_page_almost_touches_one_edge(self) -> None:
        q = np.float32([[190, 4], [675, 28], [645, 550], [145, 515]])
        self.assert_page(document(desk(600, 800, light=True), q), q)

    def test_rotated_page(self) -> None:
        q = np.float32([[380, 45], [685, 270], [420, 550], [115, 330]])
        self.assert_page(document(desk(600, 800), q), q)

    def test_perspective_page(self) -> None:
        q = np.float32([[275, 65], [700, 115], [590, 560], [95, 440]])
        self.assert_page(document(desk(600, 800, wood=True), q), q)

    def test_no_document_fails(self) -> None:
        self.assertFalse(detect_document(desk(600, 800, wood=True)).success)

    def test_light_desk_around_page_is_not_full_frame(self) -> None:
        q = np.float32([[160, 70], [680, 95], [650, 545], [125, 510]])
        image = document(desk(600, 800, light=True), q, table=True)
        self.assertFalse(detect_full_frame_page(image).success)

    def test_internal_table_alone_is_not_a_page(self) -> None:
        image = desk(600, 800, light=True)
        cv2.rectangle(image, (195, 170), (630, 460), (25, 25, 25), 4)
        for y in (230, 310, 390):
            cv2.line(image, (195, y), (630, y), (30, 30, 30), 3)
        self.assertFalse(detect_document(image).success)

    def test_rectangular_background_objects_do_not_create_page(self) -> None:
        image = desk(600, 800, wood=True)
        cv2.rectangle(image, (35, 90), (300, 275), (80, 65, 55), 5)
        cv2.rectangle(image, (485, 315), (770, 565), (75, 70, 65), 8)
        self.assertFalse(detect_document(image).success)

    def test_complex_page_prefers_outline(self) -> None:
        q = np.float32([[180, 70], [670, 105], [640, 545], [135, 510]])
        self.assert_page(document(desk(600, 800, light=True), q,
                                  table=True, complex_page=True), q)

    def test_cropped_page_can_still_use_full_frame(self) -> None:
        image = np.full((500, 350, 3), 245, np.uint8)
        for y in range(90, 380, 30):
            cv2.line(image, (50, y), (295, y), (45, 45, 45), 2)
        self.assertEqual(detect_document(image).method, "frame")

    def test_clipped_page_fails(self) -> None:
        image = np.full((500, 650, 3), 110, np.uint8)
        cv2.fillConvexPoly(image, np.int32([[375, -25], [600, 175],
                                           [300, 525], [70, 275]]), (230, 230, 230))
        result = detect_document(image)
        self.assertFalse(result.success)
        self.assertIn("four page corners", result.message)

    def test_page_clipped_at_two_opposite_edges_fails(self) -> None:
        image = document(desk(600, 800, wood=True),
                         np.float32([[430, -35], [760, 240],
                                     [355, 650], [75, 380]]))
        result = detect_document(image)
        self.assertFalse(result.success)
        self.assertNotEqual(result.method, "frame")

    def test_orb_template_extracts_features_but_rejects_random_scene(self) -> None:
        result = detect_document_orb(desk(600, 800, wood=True))
        self.assertFalse(result.success)
        self.assertNotEqual(result.message, "Template feature extraction failed.")

    def test_rotated_corner_order_is_unique(self) -> None:
        diamond = np.float32([[100, 0], [200, 100], [100, 200], [0, 100]])
        self.assertEqual(len(np.unique(order_points(diamond), axis=0)), 4)

    def test_high_resolution_resize_preserves_corners(self) -> None:
        q = np.float32([[200, 60], [700, 120], [650, 535], [160, 500]])
        image = document(desk(600, 800, wood=True), q)
        big = cv2.resize(image, (2400, 1800), interpolation=cv2.INTER_LINEAR)
        self.assert_page(big, q * 3, tolerance=30)


if __name__ == "__main__":
    unittest.main()
