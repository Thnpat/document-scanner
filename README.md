# Document Scanner

A Streamlit app that finds a flat document in a photograph, checks its four corners, corrects its perspective, and exports the result as a PNG. It supports tilted pages on textured or light backgrounds and offers several output enhancement modes.

## Features

- Automatic page boundary detection with contour analysis and existing fallbacks
- Corner overlay, perspective correction, and optional A4 output ratio
- Original color, grayscale, black and white, and sharpened output
- Image upload or camera capture, pipeline previews, and PNG download
- Light and dark themes

## How it works

```text
Upload or capture → detect page → validate four corners → rectify → enhance → download
```

The app processes a resized copy to locate the page, then maps the accepted corners back to the original image for the perspective transform. It fails when it cannot establish a reliable page boundary; it does not treat every failed detection as a full-frame scan.

### Detection methods

1. **Contours:** Canny and adaptive-threshold passes, morphological closing, contour extraction, and four-point polygon approximation. Plausible candidates are checked and ranked to favor the outer page over internal tables or boxes.
2. **Full frame:** Used only when the image borders and corners provide strong evidence that the page itself fills the photograph.
3. **ORB template matching:** An existing fallback that matches ORB features against the built-in synthetic page template and validates a RANSAC homography. The template may not match the content of an ordinary document; a sound contour detection does not depend on ORB matches.

The four corners are ordered top-left, top-right, bottom-right, bottom-left before perspective correction. A4 sizing is optional and does not constrain detection to A4 pages.

## Project structure

```text
document-scanner/
├── app.py                   Streamlit interface
├── frontend.py              Reusable UI markup and theme tokens
├── frontend.css             Responsive production styles
├── scanner/
│   ├── pipeline.py          Detection, rectification, enhancement
│   └── utils.py             Geometry and image helpers
├── tests/                  Detection and frontend workflow tests
├── .streamlit/config.toml   Theme and upload settings
├── requirements.txt         Python dependencies
└── LICENSE                  MIT license
```

## Requirements and installation

- Python 3.10 or newer
- A modern browser
- Dependencies from `requirements.txt`: Streamlit, OpenCV headless, NumPy, and Pillow

On Windows PowerShell, from the project directory:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run and use

```sh
python -m streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`. This is one Streamlit application; there is no separate backend service to start.

1. Upload a JPG, JPEG, PNG, BMP, or WebP image, or open **Use webcam instead**. The configured upload limit is 10 MB.
2. Review the selected image, choose an enhancement mode, and optionally turn off **Force A4 ratio**.
3. Click **Scan & rectify**. Review the real detected-corner overlay and rectified output, or use the failure guidance to retry or replace the photo.
4. Change the enhancement after scanning if needed, then download the PNG. **Scan another document** clears the current file and result. Expand **Technical details and pipeline images** for diagnostics.

## Input guidance and limitations

Keep all four page corners in the photo, with some visible background around the paper. Moderate rotation, perspective, texture, and shadows are supported when the page boundary remains visible. Blur, glare, severe occlusion, low page-to-background contrast, curved pages, or corners outside the photograph can prevent a reliable scan. If a page is clipped, the app reports a detection failure rather than inventing missing corners or presenting a partial rectangle as the complete page.

## Detection settings and debugging

The UI exposes the A4 output ratio before scanning and the enhancement mode before or after scanning. `run_full_pipeline()` in `scanner/pipeline.py` also accepts `max_processing_dim` (default `1500`) for the detection working image. Detection thresholds and candidate checks are in `scanner/pipeline.py`; they are not runtime UI settings.

To print detector diagnostics in the terminal, set `DOCSCAN_DEBUG=1` before starting the app. In PowerShell:

```powershell
$env:DOCSCAN_DEBUG = "1"
python -m streamlit run app.py
```

Logs include original and working dimensions, contour pass settings and counts, candidate geometry and rejection reasons, full-frame decisions, and ORB keypoint, descriptor, match, and inlier counts. The UI also keeps this scan's logs behind **Technical details and pipeline images**; developer details stay out of the primary status message.

Run the regression suite with:

```sh
python -m unittest discover -s tests -q
```

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| No page detected | Include all corners, reduce blur or glare, and increase contrast with the background. |
| An inner box appears instead of the page | Keep the entire outer edge visible; strong tables or borders can obscure a weak paper edge. |
| A page is reported as cropped | Check the original photograph at full size. A corner touching or extending beyond an image edge cannot be recovered reliably. |
| Debug log says `Insufficient good matches` | The synthetic ORB template did not match the photo. Contour detection may still succeed independently. |

## Tech stack and license

Python, Streamlit, OpenCV, NumPy, and Pillow. Licensed under the [MIT License](LICENSE).
