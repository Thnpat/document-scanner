"""Presentation helpers for the production Streamlit scanner interface."""

from __future__ import annotations

from html import escape
from pathlib import Path
import re

import streamlit as st


DARK_TOKENS = """
--ds-bg:#090d16;--ds-surface:#111726;--ds-surface-elevated:#161e31;
--ds-surface-subtle:#0d121f;--ds-border:#1e293b;--ds-border-subtle:#172033;
--ds-text-primary:#f8fafc;--ds-text-secondary:#94a3b8;--ds-text-tertiary:#64748b;
--ds-accent:#3b82f6;--ds-accent-hover:#2563eb;--ds-accent-subtle:rgba(59,130,246,.12);
--ds-accent-shadow:rgba(37,99,235,.35);--ds-accent-ring:rgba(59,130,246,.18);
--ds-on-accent:#ffffff;
--ds-success:#10b981;--ds-success-bg:rgba(16,185,129,.1);--ds-success-border:rgba(16,185,129,.25);
--ds-error:#f43f5e;--ds-error-bg:rgba(244,63,94,.1);--ds-error-border:rgba(244,63,94,.25);
--ds-badge-bg:#1e293b;--ds-doc-canvas:#06090e;
--ds-shadow-card:0 4px 20px -2px rgba(0,0,0,.5);
--ds-shadow-doc:0 8px 24px -3px rgba(0,0,0,.6);
"""

LIGHT_TOKENS = """
--ds-bg:#f8fafc;--ds-surface:#ffffff;--ds-surface-elevated:#f1f5f9;
--ds-surface-subtle:#f8fafc;--ds-border:#e2e8f0;--ds-border-subtle:#eef2f6;
--ds-text-primary:#0f172a;--ds-text-secondary:#475569;--ds-text-tertiary:#64748b;
--ds-accent:#2563eb;--ds-accent-hover:#1d4ed8;--ds-accent-subtle:rgba(37,99,235,.08);
--ds-accent-shadow:rgba(37,99,235,.28);--ds-accent-ring:rgba(37,99,235,.16);
--ds-on-accent:#ffffff;
--ds-success:#059669;--ds-success-bg:#ecfdf5;--ds-success-border:#a7f3d0;
--ds-error:#e11d48;--ds-error-bg:#fff1f2;--ds-error-border:#fecdd3;
--ds-badge-bg:#f1f5f9;--ds-doc-canvas:#e2e8f0;
--ds-shadow-card:0 2px 12px -2px rgba(0,0,0,.06),0 1px 3px rgba(0,0,0,.04);
--ds-shadow-doc:0 8px 20px -3px rgba(0,0,0,.12);
"""

SCAN_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/><path d="m7 12 5-5 5 5M12 7v10"/></svg>'
UPLOAD_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>'
CORNER_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg>'
DOCUMENT_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8"/></svg>'
BULB_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M9 18h6M10 22h4"/><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5.76.76 1.23 1.52 1.41 2.5"/></svg>'
CHECK_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true"><path d="m20 6-11 11-5-5"/></svg>'
ALERT_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="M12 8v4m0 4h.01"/></svg>'


def render_markup(fragment: str) -> None:
    """Keep custom HTML contiguous so Markdown never renders source as code."""
    cleaned = re.sub(r"<!--.*?-->", "", fragment, flags=re.DOTALL)
    lines = (line.strip() for line in cleaned.splitlines())
    st.markdown("\n".join(line for line in lines if line), unsafe_allow_html=True)


def load_styles(dark_mode: bool) -> None:
    tokens = DARK_TOKENS if dark_mode else LIGHT_TOKENS
    css = Path(__file__).with_name("frontend.css").read_text(encoding="utf-8")
    render_markup(f"<style>:root{{{tokens}}}{css}</style>")


def render_brand() -> None:
    render_markup(f"""
    <div class="app-brand">
      <span class="brand-icon-box">{SCAN_ICON}</span>
      <h1 class="brand-name">DocScan</h1>
      <span class="brand-tag">DOCUMENT SCANNER</span>
    </div>
    """)


def render_stepper(state: str) -> None:
    """Only mark a processing stage complete after the pipeline returns."""
    step_names = ("Upload", "Detect", "Rectify", "Enhance")
    completed = 4 if state == "success" else (0 if state == "empty" else 1)
    active = 0 if state == "empty" else (1 if state in ("selected", "processing", "failure") else -1)
    parts = ['<nav class="stepper-container" aria-label="Scan progress"><ol class="stepper-list">']
    for index, name in enumerate(step_names):
        status = "done" if index < completed else ("active" if index == active else "")
        if state == "failure" and index == 1:
            status = "failed"
        icon = CHECK_ICON if index < completed else str(index + 1)
        parts.append(
            f'<li class="stepper-step {status}"><span class="step-indicator-circle">'
            f'{icon}</span><span class="step-label">{name}</span></li>'
        )
        if index < len(step_names) - 1:
            parts.append('<li class="stepper-arrow" aria-hidden="true">→</li>')
    parts.append("</ol></nav>")
    render_markup("".join(parts))


def render_status(success: bool, title: str, details: tuple[str, ...] = ()) -> None:
    kind = "success" if success else "error"
    icon = CHECK_ICON if success else ALERT_ICON
    pills = "".join(f'<span class="meta-pill">{escape(item)}</span>' for item in details)
    render_markup(f"""
    <div class="status-bar status-bar-{kind}" role="status" aria-live="polite">
      <div class="status-left"><span class="status-icon-circle {kind}">{icon}</span>
      <h2 class="status-title">{escape(title)}</h2></div>
      <div class="status-meta-pills">{pills}</div>
    </div>
    """)


def render_panel_heading(title: str, badge: str = "") -> None:
    badge_html = f'<span class="canvas-badge">{escape(badge)}</span>' if badge else ""
    render_markup(f'<div class="canvas-panel-header"><h3 class="canvas-panel-title">'
                  f'{escape(title)}</h3>{badge_html}</div>')


def render_tips() -> None:
    render_markup(f"""
    <div class="tips-grid">
      <div class="tip-card"><strong>{CORNER_ICON} Complete Framing</strong><p>Keep all four corners of the page in view with visible surface margin around the perimeter.</p></div>
      <div class="tip-card"><strong>{BULB_ICON} High Surface Contrast</strong><p>Place light documents on dark or wooden tables so edge detectors can isolate boundaries cleanly.</p></div>
      <div class="tip-card"><strong>{DOCUMENT_ICON} Natural Lighting</strong><p>Ensure even, diffuse lighting to prevent hard diagonal shadows across the text.</p></div>
    </div>
    """)
