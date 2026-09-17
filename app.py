"""
Document Scanner & Perspective Rectifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Streamlit web application for CP461 — Introduction to Computer Vision.

Upload a photo of a tilted document, and this app will detect its corners
and warp it to a clean, flat top-down A4 perspective.
"""

from __future__ import annotations

import io
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from scanner.pipeline import run_full_pipeline

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocScan — Document Scanner",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp {
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Hero header ────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1a1f3a 0%, #0f1929 50%, #1a2940 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(79, 139, 249, 0.15);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(79,139,249,0.08) 0%, transparent 70%);
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}
.hero p {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0;
    max-width: 600px;
}
.hero .badge {
    display: inline-block;
    background: rgba(79, 139, 249, 0.15);
    color: #7cb3ff;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    margin-bottom: 0.75rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Cards ──────────────────────────────────────────────────────────────── */
.card {
    background: #1a1f2e;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.card:hover {
    border-color: rgba(79, 139, 249, 0.25);
}
.card h3 {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0 0 0.75rem 0;
}
.card-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

/* ── Status badges ──────────────────────────────────────────────────────── */
.status {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.35rem 0.85rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
}
.status-success {
    background: rgba(34, 197, 94, 0.12);
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.2);
}
.status-error {
    background: rgba(239, 68, 68, 0.12);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.2);
}
.status-info {
    background: rgba(79, 139, 249, 0.12);
    color: #7cb3ff;
    border: 1px solid rgba(79,139,249,0.2);
}

/* ── Metric display ─────────────────────────────────────────────────────── */
.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 0.75rem 0;
}
.metric-item {
    background: rgba(255,255,255,0.03);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    min-width: 100px;
    border: 1px solid rgba(255,255,255,0.05);
}
.metric-item .label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-item .value {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
}

/* ── Step indicator ─────────────────────────────────────────────────────── */
.steps {
    display: flex;
    gap: 0;
    margin: 1.5rem 0;
    flex-wrap: wrap;
}
.step {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    font-size: 0.82rem;
    font-weight: 500;
    color: #64748b;
    position: relative;
}
.step.active {
    color: #7cb3ff;
}
.step.done {
    color: #4ade80;
}
.step .num {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    border: 2px solid currentColor;
}
.step.done .num {
    background: #4ade80;
    color: #0f1929;
    border-color: #4ade80;
}
.step.active .num {
    background: rgba(79,139,249,0.2);
    border-color: #7cb3ff;
}
.step-arrow {
    color: #334155;
    margin: 0 0.25rem;
    font-size: 0.9rem;
}

/* ── Image containers ───────────────────────────────────────────────────── */
.img-container {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
    background: #12151f;
}
.img-container img {
    width: 100%;
    display: block;
}
.img-label {
    text-align: center;
    padding: 0.6rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: #94a3b8;
    background: rgba(255,255,255,0.02);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Mobile responsiveness ──────────────────────────────────────────────── */
@media (max-width: 768px) {
    .hero {
        padding: 1.5rem 1.25rem;
    }
    .hero h1 {
        font-size: 1.5rem;
    }
    .hero p {
        font-size: 0.92rem;
    }
    .metric-row {
        gap: 0.5rem;
    }
    .metric-item {
        min-width: 80px;
        padding: 0.4rem 0.75rem;
    }
    .steps {
        flex-direction: column;
        gap: 0.25rem;
    }
    .step-arrow {
        display: none;
    }
}

/* ── Streamlit component overrides ──────────────────────────────────────── */
.stFileUploader > div {
    border-radius: 12px !important;
}
div[data-testid="stExpander"] {
    border-radius: 12px !important;
    border-color: rgba(255,255,255,0.06) !important;
}
.stDownloadButton > button {
    width: 100%;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, #4F8BF9, #3b6fd4) !important;
    border: none !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(79,139,249,0.3) !important;
}
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="badge">CP461 · Computer Vision</div>
    <h1>📄 Document Scanner</h1>
    <p>Upload a photo of a tilted document and instantly get a clean,
       flat top-down scan — powered by ORB features, RANSAC homography,
       and adaptive edge detection.</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Step indicator
# ──────────────────────────────────────────────────────────────────────────────

def render_steps(current: int) -> None:
    """Render the pipeline step indicators. *current* is 0-based."""
    labels = ["Upload", "Detect", "Rectify", "Enhance"]
    parts = []
    for i, label in enumerate(labels):
        cls = "done" if i < current else ("active" if i == current else "")
        icon = "✓" if i < current else str(i + 1)
        parts.append(f'<span class="step {cls}"><span class="num">{icon}</span>{label}</span>')
        if i < len(labels) - 1:
            parts.append('<span class="step-arrow">→</span>')
    st.markdown(f'<div class="steps">{"".join(parts)}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_image(source) -> np.ndarray | None:
    """Read an uploaded file or camera capture into a BGR numpy array."""
    try:
        file_bytes = np.frombuffer(source.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert a BGR numpy image to a PIL Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def to_download_bytes(bgr: np.ndarray, fmt: str = "png") -> bytes:
    """Encode a BGR image to bytes for download."""
    pil = to_pil(bgr)
    buf = io.BytesIO()
    pil.save(buf, format=fmt.upper(), quality=95)
    return buf.getvalue()


PREVIEW_WIDTH = 560


# ──────────────────────────────────────────────────────────────────────────────
# Upload section
# ──────────────────────────────────────────────────────────────────────────────

render_steps(0)

upload_col, camera_col = st.columns([3, 2], gap="large")

with upload_col:
    st.markdown('<p class="card-label">📁 Upload an image</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a document photo",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

with camera_col:
    st.markdown('<p class="card-label">📸 Or use your camera</p>', unsafe_allow_html=True)
    camera = st.camera_input(
        "Take a photo",
        label_visibility="collapsed",
    )

# Pick whichever source is available (prefer new upload over camera)
source = uploaded or camera

if source is None:
    st.markdown("""
    <div class="card" style="text-align:center; padding:3rem 2rem;">
        <p style="font-size:2.5rem; margin:0;">📷</p>
        <h3 style="margin:0.5rem 0 0.25rem 0;">No image uploaded yet</h3>
        <p style="color:#64748b; font-size:0.9rem; margin:0;">
            Drag & drop a photo of a document above, or snap one with your camera.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Load image
# ──────────────────────────────────────────────────────────────────────────────

image = load_image(source)

if image is None:
    st.error("⚠️ Could not decode the image. Please try a different file.")
    st.stop()

h, w = image.shape[:2]
file_size_kb = len(source.getvalue()) / 1024

# ──────────────────────────────────────────────────────────────────────────────
# Settings sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    st.markdown("**Output Enhancement**")
    enhance_mode = st.selectbox(
        "Enhancement mode",
        options=["original", "grayscale", "bw_scan", "sharpen"],
        format_func=lambda x: {
            "original": "🖼️ Original (Color)",
            "grayscale": "🌑 Grayscale",
            "bw_scan": "📄 B&W Scanned Look",
            "sharpen": "🔍 Sharpened",
        }[x],
        label_visibility="collapsed",
    )

    st.markdown("**A4 Aspect Ratio**")
    use_a4 = st.toggle("Force A4 ratio", value=True,
                        help="Enforce standard A4 (1:√2) aspect ratio on the output")

    st.markdown("---")
    st.markdown("**Pipeline Info**")
    show_pipeline = st.toggle("Show pipeline steps", value=False,
                              help="Display intermediate CV processing stages")

# ──────────────────────────────────────────────────────────────────────────────
# Preview uploaded image
# ──────────────────────────────────────────────────────────────────────────────

render_steps(1)

st.markdown(f"""
<div class="metric-row">
    <div class="metric-item">
        <div class="label">Resolution</div>
        <div class="value">{w} × {h}</div>
    </div>
    <div class="metric-item">
        <div class="label">File Size</div>
        <div class="value">{file_size_kb:.0f} KB</div>
    </div>
    <div class="metric-item">
        <div class="label">Format</div>
        <div class="value">{source.name.rsplit('.', 1)[-1].upper() if hasattr(source, 'name') and source.name else 'CAM'}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Process
# ──────────────────────────────────────────────────────────────────────────────

scan_btn = st.button("🔍  Scan Document", use_container_width=True, type="primary")

if not scan_btn and "result" not in st.session_state:
    # Show the original as a preview
    st.image(to_pil(image), caption="Original image", width=PREVIEW_WIDTH)
    st.stop()

# Run the pipeline (or reuse cached result)
if scan_btn:
    with st.spinner("Detecting document and computing perspective transform…"):
        t0 = time.perf_counter()
        result = run_full_pipeline(
            image,
            enhance_mode=enhance_mode,
            use_a4_ratio=use_a4,
        )
        elapsed = time.perf_counter() - t0
        st.session_state["result"] = result
        st.session_state["elapsed"] = elapsed
        st.session_state["enhance_mode"] = enhance_mode
else:
    result = st.session_state["result"]
    elapsed = st.session_state.get("elapsed", 0)
    # Re-apply enhancement if mode changed
    if enhance_mode != st.session_state.get("enhance_mode") and result.success and result.warped is not None:
        from scanner.pipeline import enhance_scan
        result.enhanced = enhance_scan(result.warped, mode=enhance_mode)
        st.session_state["enhance_mode"] = enhance_mode

# ──────────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────────

if not result.success:
    render_steps(1)

    st.markdown(f"""
    <div class="status status-error">⚠ Detection failed</div>
    """, unsafe_allow_html=True)

    st.warning(f"**{result.message}**")

    st.markdown("""
    <div class="card">
        <h3>💡 Tips to improve detection</h3>
        <ul style="color:#94a3b8; font-size:0.9rem; line-height:1.8;">
            <li>Place the document on a <strong>contrasting background</strong> (dark desk, colored mat).</li>
            <li>Ensure <strong>even lighting</strong> — avoid harsh shadows across the page.</li>
            <li>Keep the <strong>entire document visible</strong> within the frame.</li>
            <li>Flatten the paper to minimize <strong>curling or folding</strong>.</li>
            <li>Avoid <strong>cluttered backgrounds</strong> with many edges.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Still show original
    st.image(to_pil(image), caption="Original image", width=PREVIEW_WIDTH)

else:
    render_steps(4)

    st.markdown(f"""
    <div class="status status-success">✓ Document scanned successfully</div>
    """, unsafe_allow_html=True)

    method_label = {
        "contour": "Contour", "frame": "Full frame", "orb": "ORB",
    }.get(result.detection.method if result.detection else "", "Unknown")
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-item">
            <div class="label">Method</div>
            <div class="value">{method_label}</div>
        </div>
        <div class="metric-item">
            <div class="label">Time</div>
            <div class="value">{elapsed:.2f}s</div>
        </div>
        <div class="metric-item">
            <div class="label">Output</div>
            <div class="value">{result.enhanced.shape[1]}×{result.enhanced.shape[0]}</div>
        </div>
        {"" if not result.detection or result.detection.orb_matches == 0 else f'''
        <div class="metric-item">
            <div class="label">ORB Matches</div>
            <div class="value">{result.detection.orb_matches}</div>
        </div>
        '''}
    </div>
    """, unsafe_allow_html=True)

    # ── Side-by-side comparison ──────────────────────────────────────────
    col_orig, col_result = st.columns(2, gap="medium")

    with col_orig:
        st.markdown('<div class="img-label">📷 Detected Corners</div>', unsafe_allow_html=True)
        st.image(to_pil(result.corners_overlay), width=PREVIEW_WIDTH)

    with col_result:
        mode_labels = {
            "original": "🖼️ Rectified",
            "grayscale": "🌑 Grayscale",
            "bw_scan": "📄 B&W Scan",
            "sharpen": "🔍 Sharpened",
        }
        label = mode_labels.get(enhance_mode, "Rectified")
        st.markdown(f'<div class="img-label">{label} Output</div>', unsafe_allow_html=True)
        st.image(to_pil(result.enhanced), width=PREVIEW_WIDTH)

    # ── Download ─────────────────────────────────────────────────────────
    st.markdown("")
    dl_col1, dl_col2, dl_col3 = st.columns([1, 2, 1])
    with dl_col2:
        st.download_button(
            label="⬇️  Download Scanned Document",
            data=to_download_bytes(result.enhanced),
            file_name="scanned_document.png",
            mime="image/png",
            use_container_width=True,
        )

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline visualization (debug)
# ──────────────────────────────────────────────────────────────────────────────

if show_pipeline and result.detection and result.detection.debug_images:
    st.markdown("---")
    st.markdown("### 🔬 Pipeline Visualization")
    st.caption("Intermediate processing stages for educational / debugging purposes.")

    debug_imgs = result.detection.debug_images
    step_names = {
        "grayscale": "Grayscale Conversion",
        "edges_auto": "Auto Canny Edges",
        "edges_morphed": "Morphological Closing",
        "edges_tight": "Tight Canny Edges",
        "adaptive_threshold": "Adaptive Threshold",
        "corners_detected": "Corners Detected",
        "orb_keypoints": "ORB Keypoints",
        "orb_corners_detected": "ORB-based Corners",
    }

    # Render in rows of 2-3
    keys = list(debug_imgs.keys())
    for i in range(0, len(keys), 3):
        chunk = keys[i:i + 3]
        cols = st.columns(len(chunk), gap="medium")
        for col, key in zip(cols, chunk):
            with col:
                nice_name = step_names.get(key, key.replace("_", " ").title())
                st.image(to_pil(debug_imgs[key]), caption=nice_name, width=PREVIEW_WIDTH)

    # Show homography matrix
    if result.homography is not None:
        with st.expander("📐 Homography Matrix (3×3)"):
            st.code(np.array2string(result.homography, precision=4, suppress_small=True))

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#475569; font-size:0.8rem;">'
    'Built with OpenCV, Streamlit & ❤️ for CP461 — Introduction to Computer Vision'
    '</p>',
    unsafe_allow_html=True,
)
