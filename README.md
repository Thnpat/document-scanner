# 📄 Document Scanner & Perspective Rectifier

A computer-vision web application that detects tilted paper corners in a photograph and rectifies the document to a clean, flat **top-down A4 perspective**.

Built for **CP461 — Introduction to Computer Vision** (Semester 1/2026).

---

## ✨ Features

- **Document Detection** — Contour-based detection, support for pages that already fill the frame, and ORB fallback
- **RANSAC Homography** — Robust geometric estimation that handles outliers
- **Sub-pixel Corner Refinement** — `cv2.cornerSubPix` for precise corner localization
- **A4 Perspective Rectification** — Warps detected documents to standard A4 aspect ratio
- **Post-processing Modes** — Original color, grayscale, B&W scanned look, or sharpened
- **Pipeline Visualization** — View intermediate processing stages (edges, contours, keypoints)
- **Mobile Responsive** — Camera capture support + responsive layout for phone use
- **Light & Dark Mode** — Switch themes with smooth color transitions; the selected mode survives page refreshes
- **One-click Download** — Export the scanned document as PNG

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| CV Pipeline | OpenCV (Canny, ORB, findHomography, warpPerspective) |
| Web Framework | Streamlit |
| Image Processing | NumPy, Pillow |
| Deployment | Streamlit Cloud |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+

### Install & Run

```bash
# Clone the repository
git clone https://github.com/Thnpat/document-scanner.git
cd document-scanner

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

### Deploy to Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) and connect GitHub.
2. Create an app from the `Thnpat/document-scanner` repository.
3. Select branch `main` and set the entrypoint file to `app.py`.
4. Deploy. The root `requirements.txt` and `.streamlit/config.toml` are used automatically.

## 📁 Project Structure

```
document-scanner/
├── .streamlit/
│   └── config.toml          # Streamlit theme & server config
├── scanner/
│   ├── __init__.py           # Package exports
│   ├── pipeline.py           # Core CV pipeline (detect → rectify → enhance)
│   └── utils.py              # Geometry & image helpers
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
└── README.md
```

## 🔬 Pipeline Overview

```
Input Image
    │
    ▼
Pre-processing (grayscale → Gaussian blur → bilateral filter)
    │
    ▼
Edge Detection (auto-Canny → morphological closing)
    │
    ├── Success? → Contour Extraction → 4-point polygon
    │                                        │
    │                                        ▼
    │                              Corner Sub-pixel Refinement
    │                                        │
    └── Failure? → ORB Keypoints ──────┐     │
                   + BFMatcher         │     │
                   + Ratio Test        │     │
                        │              │     │
                        ▼              │     │
                   Homography (RANSAC) │     │
                        │              │     │
                        ▼              ▼     ▼
                   Template Corner     Ordered Corners [TL, TR, BR, BL]
                   Mapping                   │
                                             ▼
                                   Perspective Warp (A4)
                                             │
                                             ▼
                                   Post-processing (enhance)
                                             │
                                             ▼
                                        Output Image
```

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
