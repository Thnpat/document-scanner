"""Production Streamlit interface for the document scanner."""

from __future__ import annotations

from html import escape
import base64
import io
import logging
import os
import time

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from frontend import (BULB_ICON, CORNER_ICON, DOCUMENT_ICON, UPLOAD_ICON,
                      load_styles, render_brand, render_markup,
                      render_panel_heading, render_status, render_stepper,
                      render_tips)
from scanner.pipeline import enhance_scan, run_full_pipeline


st.set_page_config(page_title="DocScan — Document Scanner", page_icon="📄",
                   layout="wide", initial_sidebar_state="collapsed")

if os.getenv("DOCSCAN_DEBUG") == "1":
    scanner_logger = logging.getLogger("scanner.pipeline")
    scanner_logger.setLevel(logging.DEBUG)
    if not scanner_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        scanner_logger.addHandler(console_handler)


ENHANCEMENT_LABELS = {
    "bw_scan": "B&W Scan",
    "original": "Color",
    "grayscale": "Grayscale",
    "sharpen": "Sharpened",
}
METHOD_LABELS = {"contour": "Contour", "frame": "Full frame", "orb": "ORB"}


def initialize_state() -> None:
    defaults = {
        "dark_mode": st.query_params.get("theme", "dark") != "light",
        "source_bytes": None,
        "source_name": "",
        "source_type": "",
        "picker_open": False,
        "show_camera": False,
        "picker_generation": 0,
        "result": None,
        "scan_error": None,
        "elapsed": None,
        "diagnostics": "",
        "enhance_mode": "bw_scan",
        "rendered_mode": "bw_scan",
        "use_a4_ratio": True,
        "is_processing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_theme() -> None:
    st.query_params["theme"] = "dark" if st.session_state.dark_mode else "light"


def reset_scan() -> None:
    """Increment widget keys so an earlier upload cannot reappear on rerun."""
    st.session_state.source_bytes = None
    st.session_state.source_name = ""
    st.session_state.source_type = ""
    st.session_state.result = None
    st.session_state.scan_error = None
    st.session_state.elapsed = None
    st.session_state.diagnostics = ""
    st.session_state.enhance_mode = "bw_scan"
    st.session_state.rendered_mode = "bw_scan"
    st.session_state.use_a4_ratio = True
    st.session_state.is_processing = False
    st.session_state.picker_open = False
    st.session_state.show_camera = False
    st.session_state.picker_generation += 1


def open_picker() -> None:
    st.session_state.picker_open = True
    st.session_state.show_camera = False
    st.session_state.picker_generation += 1


def toggle_camera() -> None:
    st.session_state.show_camera = not st.session_state.show_camera


def request_scan() -> None:
    st.session_state.is_processing = True


def load_image(data: bytes | None) -> np.ndarray | None:
    if not data:
        return None
    try:
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return None


def to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def preview_image(image: np.ndarray) -> Image.Image:
    """Bound display size without changing the scan or downloaded output."""
    preview = to_pil(image)
    preview.thumbnail((720, 440), Image.Resampling.LANCZOS)
    return preview


def to_download_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    to_pil(image).save(buffer, format="PNG")
    return buffer.getvalue()


def thumbnail_uri(image: np.ndarray) -> str:
    thumbnail = to_pil(image)
    thumbnail.thumbnail((96, 96), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def render_header() -> None:
    with st.container(key="app_header"):
        brand_col, theme_col = st.columns([4, 1], vertical_alignment="center")
        with brand_col:
            render_brand()
        with theme_col:
            st.toggle("Dark mode", key="dark_mode", on_change=save_theme)


def render_picker() -> None:
    with st.container(key="workspace_card"):
        with st.container(key="upload_dropzone"):
            render_markup(f"""
            <div class="dropzone-icon">{UPLOAD_ICON}</div>
            <h2 class="dropzone-title">Upload a document photo</h2>
            <p class="dropzone-subtitle">Drag and drop your file here, or click to browse</p>
            <div class="format-pills">
              <span class="format-pill">JPG</span><span class="format-pill">PNG</span>
              <span class="format-pill">WEBP</span><span class="format-pill">BMP</span>
              <span class="format-pill">MAX 10MB</span>
            </div>
            """)
            generation = st.session_state.picker_generation
            with st.container(key="upload_actions"):
                uploaded = st.file_uploader("Browse Files",
                                            type=["jpg", "jpeg", "png", "bmp", "webp"],
                                            key=f"upload_{generation}",
                                            label_visibility="collapsed")
                st.button("Use Webcam", key="webcam_action", icon=":material/photo_camera:",
                          on_click=toggle_camera)
            camera = None
            if st.session_state.show_camera:
                with st.container(key="camera_area"):
                    camera = st.camera_input("Take a document photo", key=f"camera_{generation}")
            source = uploaded or camera
            if source is not None:
                st.session_state.source_bytes = source.getvalue()
                st.session_state.source_name = source.name or "Camera photo"
                st.session_state.source_type = source.type or "image/jpeg"
                st.session_state.result = None
                st.session_state.scan_error = None
                st.session_state.elapsed = None
                st.session_state.diagnostics = ""
                st.session_state.picker_open = False
                st.session_state.show_camera = False
                st.rerun()
    if st.session_state.source_bytes is not None:
        st.button("Cancel replacement", on_click=lambda: setattr(st.session_state, "picker_open", False))


def render_selected_summary(image: np.ndarray) -> None:
    height, width = image.shape[:2]
    size_kb = len(st.session_state.source_bytes) / 1024
    file_name = escape(st.session_state.source_name)
    file_type = escape(st.session_state.source_type)
    thumbnail = thumbnail_uri(image)
    with st.container(key="selected_card"):
        render_markup(f"""
        <div class="selected-summary"><div class="selected-file">
          <img class="selected-thumbnail" src="{thumbnail}" alt="Selected document thumbnail"/><div>
          <strong>{file_name}</strong>
          <p>{width} × {height} px &nbsp;·&nbsp; {size_kb:.0f} KB &nbsp;·&nbsp; {file_type}</p>
        </div></div><span class="meta-pill selected-ready">Ready to scan</span></div>
        """)


def render_image_panel(key: str, image: np.ndarray, title: str, badge: str = "",
                       footnote: str = "") -> None:
    with st.container(key=key):
        render_panel_heading(title, badge)
        st.image(preview_image(image), width="content")
        if footnote:
            render_markup(f'<div class="panel-footnote">{escape(footnote)}</div>')


def render_filter_controls() -> str:
    return st.radio("Enhancement", tuple(ENHANCEMENT_LABELS),
                    format_func=lambda value: ENHANCEMENT_LABELS[value],
                    key="enhance_mode", horizontal=True)


def scan_image(image: np.ndarray) -> None:
    """Run the existing synchronous pipeline once and retain its real diagnostics."""
    with st.container(key="processing_card"):
        render_markup("""
        <div class="processing-body" role="status" aria-live="polite">
          <div class="processing-icon" aria-hidden="true">◌</div>
          <h2>Rectifying Document Perspective...</h2>
          <p>Analyzing the page boundary and calculating its perspective transform. Please wait.</p>
        </div>
        """)
        diagnostic_buffer = io.StringIO()
        pipeline_logger = logging.getLogger("scanner.pipeline")
        previous_level = pipeline_logger.level
        log_handler = logging.StreamHandler(diagnostic_buffer)
        log_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        pipeline_logger.addHandler(log_handler)
        pipeline_logger.setLevel(logging.DEBUG)
        try:
            with st.spinner("Processing the selected image…"):
                started = time.perf_counter()
                result = run_full_pipeline(
                    image,
                    enhance_mode=st.session_state.enhance_mode,
                    use_a4_ratio=st.session_state.use_a4_ratio,
                )
                elapsed = time.perf_counter() - started
            st.session_state.result = result
            st.session_state.elapsed = elapsed
            st.session_state.rendered_mode = st.session_state.enhance_mode
            st.session_state.diagnostics = diagnostic_buffer.getvalue()
            st.session_state.scan_error = None
        except Exception as exc:
            st.session_state.result = None
            st.session_state.scan_error = "The scan could not be completed. Try another image."
            st.session_state.diagnostics = (
                diagnostic_buffer.getvalue() +
                f"\nProcessing error: {type(exc).__name__}: {exc}"
            )
        finally:
            pipeline_logger.removeHandler(log_handler)
            pipeline_logger.setLevel(previous_level)
            st.session_state.is_processing = False
    st.rerun()


def render_selected(image: np.ndarray) -> None:
    render_selected_summary(image)
    render_image_panel("source_panel", image, "Source document preview", "ORIGINAL RESOLUTION")
    with st.container(key="controls_ribbon"):
        options_col, actions_col = st.columns([3, 2], vertical_alignment="bottom")
        with options_col:
            render_filter_controls()
            st.toggle("Force A4 ratio", value=True, key="use_a4_ratio",
                      help="Set the rectified output to A4 proportions")
        with actions_col:
            replace_col, scan_col = st.columns(2)
            with replace_col:
                st.button("Choose different", on_click=open_picker, width="stretch")
            with scan_col:
                st.button("Scan & rectify", type="primary", on_click=request_scan,
                          width="stretch", disabled=st.session_state.is_processing)


def render_technical_details(result) -> None:
    with st.container(key="technical_details"):
        with st.expander("Technical Diagnostics (OpenCV Candidates & ORB Homography)"):
            if result is not None and result.detection is not None:
                detection = result.detection
                render_markup(f"""
                <div class="technical-summary">
                  <div><strong>Detector</strong><span>{escape(detection.method or 'none')}</span></div>
                  <div><strong>ORB ratio-test matches</strong><span>{detection.orb_matches}</span></div>
                  <div><strong>Detection message</strong><span>{escape(detection.message)}</span></div>
                </div>
                """)
            if st.session_state.diagnostics:
                lines = escape(st.session_state.diagnostics).replace("\n", "<br>")
                render_markup(f'<div class="diagnostic-log">{lines}</div>')
            if result is not None and result.detection is not None:
                for name, debug_image in result.detection.debug_images.items():
                    st.image(preview_image(debug_image),
                             caption=name.replace("_", " ").title(), width="content")
            if result is not None and result.homography is not None:
                render_markup('<p class="technical-heading">Homography matrix</p>')
                matrix = np.array2string(result.homography, precision=4, suppress_small=True)
                lines = escape(matrix).replace("\n", "<br>")
                render_markup(f'<div class="diagnostic-log diagnostic-matrix">{lines}</div>')


def render_success(image: np.ndarray, result) -> None:
    selected_mode = st.session_state.enhance_mode
    if selected_mode != st.session_state.rendered_mode and result.warped is not None:
        result.enhanced = enhance_scan(result.warped, mode=selected_mode)
        st.session_state.rendered_mode = selected_mode

    output = result.enhanced
    method = METHOD_LABELS.get(result.detection.method, result.detection.method.title())
    elapsed = st.session_state.elapsed
    render_status(True, "Document Rectified Successfully",
                  (f"Method: {method}", f"Time: {elapsed:.2f}s",
                   f"Output: {output.shape[1]} × {output.shape[0]} px"))

    with st.container(key="result_grid"):
        left, right = st.columns(2, gap="medium")
        with left:
            corner_note = ""
            if result.detection.corners is not None:
                scale = min(1.0, 1500 / max(image.shape[:2]))
                points = np.rint(result.detection.corners / scale).astype(int)
                labels = ("TL", "TR", "BR", "BL")
                corner_note = "   ".join(
                    f"{label}: ({point[0]}, {point[1]})"
                    for label, point in zip(labels, points)
                )
            render_image_panel("detected_panel", result.corners_overlay,
                               "Detected Page Boundaries", "4 CORNERS VERIFIED", corner_note)
        with right:
            ratio_note = "A4 ratio enforced" if st.session_state.use_a4_ratio else "Original page ratio"
            render_image_panel("rectified_panel", output,
                               f"Rectified Output ({ENHANCEMENT_LABELS[selected_mode]})",
                               "A4 · TOP-DOWN" if st.session_state.use_a4_ratio else "PAGE RATIO",
                               ratio_note)

    with st.container(key="controls_ribbon"):
        filter_col, action_col = st.columns([2, 3], vertical_alignment="bottom")
        with filter_col:
            render_filter_controls()
        with action_col:
            action_left, action_right = st.columns([1, 2])
            with action_left:
                st.button("Scan another document", on_click=reset_scan, width="stretch")
            with action_right:
                st.download_button("Download scanned document (.PNG)",
                                   data=to_download_bytes(output),
                                   file_name="scanned_document.png", mime="image/png",
                                   type="primary", width="stretch")
    render_technical_details(result)


def render_failure(image: np.ndarray | None, result) -> None:
    render_status(False, "Unable to Locate Document Boundaries")
    message = (result.message if result is not None else
               st.session_state.scan_error or "Could not read this image file.")
    with st.container(key="failure_grid"):
        image_col, advice_col = st.columns(2, gap="medium")
        with image_col:
            if image is not None:
                render_image_panel("source_panel", image, "Uploaded Image", "ORIGINAL PHOTO")
        with advice_col:
            with st.container(key="guidance_card"):
                render_markup(f"""
                <h3>How to improve scan detection</h3>
                <p class="failure-intro">{escape(message)}</p>
                <div class="tip-card"><strong>{CORNER_ICON} Keep margins around all 4 corners</strong><p>Leave some table surface visible on every side of the paper.</p></div>
                <div class="tip-card"><strong>{BULB_ICON} Use a contrasting surface</strong><p>Separate the paper from the desk, especially near its edges.</p></div>
                <div class="tip-card"><strong>{DOCUMENT_ICON} Avoid severe tilt and glare</strong><p>Hold the camera above the document in even, diffuse light.</p></div>
                """)
                retry_col, replace_col = st.columns(2)
                with retry_col:
                    st.button("Retry scan", type="primary", on_click=request_scan,
                              disabled=image is None, width="stretch")
                with replace_col:
                    st.button("Choose another photo", on_click=open_picker, width="stretch")
    render_technical_details(result)


initialize_state()
load_styles(st.session_state.dark_mode)
render_header()

data = st.session_state.source_bytes
image = load_image(data)
result = st.session_state.result

if data is None:
    state = "empty"
elif image is None:
    state = "failure"
elif st.session_state.is_processing:
    state = "processing"
elif st.session_state.scan_error:
    state = "failure"
elif result is None:
    state = "selected"
else:
    state = "success" if result.success else "failure"

render_stepper(state)

if data is None or st.session_state.picker_open:
    render_picker()
    if data is None:
        render_tips()
        st.stop()

if state == "selected":
    render_selected(image)
elif state == "processing":
    scan_image(image)
elif state == "success":
    render_success(image, result)
elif state == "failure":
    render_failure(image, result)
