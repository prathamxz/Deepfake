"""
DeepGuard AI — Premium Web Interface v2
Standalone web app for image & video deepfake detection.
Uses the same trained model as the Chrome extension.

Run:  streamlit run app.py
"""

import os
import io
import time
import tempfile
import requests
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from streamlit_lottie import st_lottie

# ── Path setup ──────────────────────────────────────────────────────────────
from src.config import IMG_SIZE, BATCH_SIZE, MODEL_PATH, MODEL_PATH_H5, FACE_PADDING, FRAME_STEP, MODELS_DIR
from src.model import build_model, CUSTOM_OBJECTS

# ── Decision Threshold ──────────────────────────────────────────────────────
DECISION_THRESHOLD = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# Lottie Animation Loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_lottie_url(url: str):
    """Fetches a Lottie JSON animation from a URL. Returns None on failure."""
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


# Premium vector animations (loaded once, cached)
LOTTIE_SHIELD = load_lottie_url("https://lottie.host/8099307d-dc77-44a6-89ba-8d26732efcfc/JmZcshvU2G.json")
LOTTIE_PROCESSING = load_lottie_url("https://lottie.host/e7d7db31-50be-4b95-bf3a-96ce19999ffb/NnOnz6A2pW.json")

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepGuard AI — Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Persist history & analysis count across reruns
if "history" not in st.session_state:
    st.session_state.history = []
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0
if "img_uploader_version" not in st.session_state:
    st.session_state.img_uploader_version = 0
if "vid_uploader_version" not in st.session_state:
    st.session_state.vid_uploader_version = 0
if "cam_uploader_version" not in st.session_state:
    st.session_state.cam_uploader_version = 0

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Premium Dark UI with Glassmorphism + Animations
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Animated Cyber Glow Background ────────── */
    .stApp {
        background: linear-gradient(-45deg, #070714, #0f0c20, #0b1528, #05050d);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Floating particle layer */
    .particles {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    .particle {
        position: absolute;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(108,99,255,0.35), transparent 70%);
        animation: floatUp linear infinite;
    }
    @keyframes floatUp {
        0%   { transform: translateY(110vh) translateX(0) scale(0.6); opacity: 0; }
        10%  { opacity: 0.7; }
        90%  { opacity: 0.4; }
        100% { transform: translateY(-10vh) translateX(40px) scale(1.2); opacity: 0; }
    }

    /* ── Fade / slide-in entrance for content ──── */
    .fade-in { animation: fadeIn 0.7s ease both; }
    .slide-up { animation: slideUp 0.6s cubic-bezier(.2,.8,.2,1) both; }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes slideUp { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: translateY(0); } }

    /* ── Hero Header ───────────────────────────── */
    .hero {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
        position: relative;
        z-index: 1;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -10%;
        left: 25%;
        width: 380px;
        height: 380px;
        background: radial-gradient(circle, rgba(108,99,255,0.25) 0%, transparent 70%);
        filter: blur(40px);
        pointer-events: none;
        z-index: 0;
        animation: floatingGlow 8s ease-in-out infinite;
    }
    @keyframes floatingGlow {
        0%, 100% { transform: translate(-50%, 0) scale(1); opacity: 0.7; }
        50% { transform: translate(-30%, -20px) scale(1.15); opacity: 1; }
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(108,99,255,0.2), rgba(72,202,228,0.2));
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 100px;
        padding: 0.35rem 1.2rem;
        font-size: 0.78rem;
        font-weight: 600;
        color: #8b8bff;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        animation: fadeIn 1s ease;
    }
    .hero-badge .dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #38ef7d;
        margin-right: 6px;
        box-shadow: 0 0 8px #38ef7d;
        animation: blink 1.6s ease-in-out infinite;
    }
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
    .hero h1 {
        font-size: 3.4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #6C63FF 0%, #48CAE4 50%, #38ef7d 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
        animation: shimmer 5s linear infinite, slideUp 0.7s ease both;
    }
    @keyframes shimmer { to { background-position: 200% center; } }
    .hero p {
        color: #7a7a9e;
        font-size: 1.05rem;
        max-width: 620px;
        margin: 0 auto;
        line-height: 1.6;
        animation: fadeIn 1.2s ease;
    }

    /* ── Sparkle Burst (Real verdict) ───────────── */
    .sparkle-burst {
        position: relative;
        width: 100%;
        height: 0;
        pointer-events: none;
    }
    .sparkle {
        position: absolute;
        top: 0; left: 50%;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #38ef7d;
        box-shadow: 0 0 8px #38ef7d, 0 0 16px rgba(56,239,125,0.6);
        opacity: 0;
        animation: sparkleFly 1s ease-out forwards;
    }
    @keyframes sparkleFly {
        0%   { opacity: 1; transform: translate(-50%, 0) scale(1); }
        100% { opacity: 0; transform: translate(var(--dx), var(--dy)) scale(0.3); }
    }

    /* ── Drag & Drop Highlight ──────────────────── */
    div[data-testid="stFileUploaderDropzone"] {
        transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1);
        border-radius: 14px !important;
    }
    div[data-testid="stFileUploaderDropzone"]:has(input:focus),
    div[data-testid="stFileUploaderDropzone"].dragover {
        border-color: #6C63FF !important;
        background: rgba(108,99,255,0.08) !important;
        box-shadow: 0 0 0 3px rgba(108,99,255,0.15);
    }

    /* ── Streak Badge ───────────────────────────── */
    .streak-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: linear-gradient(135deg, rgba(255,165,0,0.15), rgba(255,200,0,0.08));
        border: 1px solid rgba(255,200,0,0.3);
        border-radius: 100px;
        padding: 0.4rem 1rem;
        font-size: 0.8rem;
        color: #ffd166;
        font-weight: 700;
        animation: fadeIn 0.5s ease both;
    }

    /* ── Flip Cards (How it Works) ─────────────── */
    .flip-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 1rem;
        max-width: 980px;
        margin: 2rem auto 0.5rem;
        perspective: 1000px;
    }
    .flip-card {
        height: 140px;
        cursor: pointer;
    }
    .flip-inner {
        position: relative;
        width: 100%;
        height: 100%;
        transition: transform 0.6s cubic-bezier(0.25, 0.8, 0.25, 1);
        transform-style: preserve-3d;
    }
    .flip-card:hover .flip-inner {
        transform: rotateY(180deg);
    }
    .flip-face {
        position: absolute;
        width: 100%;
        height: 100%;
        border-radius: 16px;
        backface-visibility: hidden;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .flip-front {
        background: rgba(255,255,255,0.03);
    }
    .flip-back {
        background: linear-gradient(135deg, rgba(108,99,255,0.18), rgba(72,202,228,0.12));
        transform: rotateY(180deg);
        border-color: rgba(108,99,255,0.3);
    }
    .flip-icon { font-size: 2rem; margin-bottom: 0.4rem; }
    .flip-title { color: #e0e0ff; font-weight: 700; font-size: 0.92rem; }
    .flip-back .flip-title { font-size: 0.85rem; margin-bottom: 0.4rem; }
    .flip-desc { color: #b0b0d0; font-size: 0.76rem; line-height: 1.4; }

    /* ── Glass Card ────────────────────────────── */
    .glass-card {
        background: rgba(255,255,255,0.02);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
        will-change: transform;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        background: rgba(255,255,255,0.04);
        border-color: rgba(108,99,255,0.25);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4), 0 0 20px rgba(108,99,255,0.05);
    }
    .glass-card-sm {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
    }

    /* ── Verdict Banners ───────────────────────── */
    .verdict-fake {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 900;
        letter-spacing: 0.03em;
        box-shadow: 0 8px 40px rgba(255, 65, 108, 0.35);
        animation: pulse-red 2s infinite, slideUp 0.5s ease both;
    }
    .verdict-real {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 900;
        letter-spacing: 0.03em;
        box-shadow: 0 8px 40px rgba(17, 153, 142, 0.35);
        animation: pulse-green 2.4s infinite, slideUp 0.5s ease both;
    }
    @keyframes pulse-red {
        0%, 100% { transform: scale(1); box-shadow: 0 8px 40px rgba(255, 65, 108, 0.35); }
        50% { transform: scale(1.01); box-shadow: 0 12px 50px rgba(255, 65, 108, 0.5); }
    }
    @keyframes pulse-green {
        0%, 100% { transform: scale(1); box-shadow: 0 8px 40px rgba(17, 153, 142, 0.35); }
        50% { transform: scale(1.01); box-shadow: 0 12px 50px rgba(56, 239, 125, 0.45); }
    }

    /* ── Metric Cards ──────────────────────────── */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.25s ease, border-color 0.25s ease;
        animation: slideUp 0.5s ease both;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(108,99,255,0.35);
    }
    .metric-label { color: #7a7a9e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
    .metric-value { color: #e0e0ff; font-size: 1.8rem; font-weight: 800; margin-top: 0.3rem; }
    .metric-sub { color: #5a5a7e; font-size: 0.75rem; margin-top: 0.2rem; }

    /* ── Score Bar ──────────────────────────────── */
    .score-container {
        width: 100%;
        position: relative;
        margin: 1.5rem 0;
    }
    .score-bar-bg {
        width: 100%;
        height: 14px;
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        overflow: hidden;
    }
    .score-bar-fill-fake {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        background-size: 200% 100%;
        animation: barGrow 1s ease, gradientMove 3s linear infinite;
    }
    .score-bar-fill-real {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #11998e, #38ef7d);
        background-size: 200% 100%;
        animation: barGrow 1s ease, gradientMove 3s linear infinite;
    }
    @keyframes barGrow { from { width: 0%; } }
    @keyframes gradientMove { to { background-position: -200% 0; } }
    .score-label {
        display: flex;
        justify-content: space-between;
        color: #5a5a7e;
        font-size: 0.75rem;
        margin-top: 0.4rem;
    }

    /* ── Confidence Ring ───────────────────────── */
    .ring-wrap { display: flex; justify-content: center; align-items: center; margin: 0.5rem 0; }

    /* ── Upload Zone ───────────────────────────── */
    .upload-zone {
        border: 2px dashed rgba(108,99,255,0.25);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        background: rgba(108,99,255,0.02);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .upload-zone:hover {
        border-color: rgba(108,99,255,0.6);
        background: rgba(108,99,255,0.06);
        transform: translateY(-3px);
    }
    .upload-icon { font-size: 3rem; margin-bottom: 0.5rem; animation: floatIcon 3s ease-in-out infinite; }
    @keyframes floatIcon { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
    .upload-title { color: #b0b0d0; font-size: 1.1rem; font-weight: 600; }
    .upload-sub { color: #5a5a7e; font-size: 0.85rem; margin-top: 0.3rem; }

    /* ── Warning Box ───────────────────────────── */
    .warning-box {
        background: rgba(255, 200, 0, 0.08);
        border-left: 4px solid #ffc800;
        border-radius: 0 12px 12px 0;
        padding: 0.9rem 1.2rem;
        color: #d4a800;
        font-size: 0.85rem;
        margin: 0.8rem 0;
        animation: slideUp 0.4s ease both;
    }

    /* ── Badge chips ──────────────────────────── */
    .chip {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin: 0 0.25rem 0.25rem 0;
    }
    .chip-fake { background: rgba(255,65,108,0.15); color: #ff7a93; border: 1px solid rgba(255,65,108,0.3); }
    .chip-real { background: rgba(56,239,125,0.15); color: #6dffb0; border: 1px solid rgba(56,239,125,0.3); }
    .chip-neutral { background: rgba(108,99,255,0.15); color: #a8a3ff; border: 1px solid rgba(108,99,255,0.3); }

    /* ── History row ──────────────────────────── */
    .history-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0.9rem;
        border-radius: 10px;
        background: rgba(255,255,255,0.03);
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
        color: #b0b0d0;
        animation: slideUp 0.4s ease both;
    }

    /* ── Footer ────────────────────────────────── */
    .footer {
        text-align: center;
        color: #3a3a5e;
        font-size: 0.78rem;
        padding: 3rem 0 1rem;
        border-top: 1px solid rgba(255,255,255,0.04);
        margin-top: 3rem;
    }
    .footer .heart { color: #ff416c; animation: pulse-red 1.6s infinite; display: inline-block; }

    /* ── Powered By Strip ───────────────────────── */
    .powered-by {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.04);
    }
    .powered-by-label {
        color: #4a4a6e;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-right: 0.4rem;
    }
    .tech-badge {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 600;
        color: #5a5a7e;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        filter: grayscale(1);
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
        cursor: default;
    }
    .tech-badge:hover {
        filter: grayscale(0);
        color: #a8a3ff;
        border-color: rgba(108,99,255,0.4);
        background: rgba(108,99,255,0.08);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(108,99,255,0.2);
    }

    /* ── Hide Streamlit Defaults ────────────────── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    div[data-testid="stProgress"] > div > div {
        border-radius: 8px !important;
        background: linear-gradient(90deg, #6C63FF, #48CAE4) !important;
        background-size: 200% 100% !important;
        animation: gradientMove 2s linear infinite !important;
    }

    /* Buttons */
    div.stButton > button {
        transition: all 0.25s ease;
        border-radius: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(108,99,255,0.35);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.06);
        padding: 0.4rem 1.2rem;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(108,99,255,0.25), rgba(72,202,228,0.18)) !important;
        border-color: rgba(108,99,255,0.4) !important;
    }
</style>

<div class="particles">
    <div class="particle" style="width:60px; height:60px; left:8%;  animation-duration: 22s; animation-delay: 0s;"></div>
    <div class="particle" style="width:40px; height:40px; left:22%; animation-duration: 28s; animation-delay: 4s;"></div>
    <div class="particle" style="width:80px; height:80px; left:48%; animation-duration: 26s; animation-delay: 2s;"></div>
    <div class="particle" style="width:50px; height:50px; left:68%; animation-duration: 32s; animation-delay: 6s;"></div>
    <div class="particle" style="width:35px; height:35px; left:85%; animation-duration: 24s; animation-delay: 1s;"></div>
    <div class="particle" style="width:65px; height:65px; left:36%; animation-duration: 30s; animation-delay: 8s;"></div>
</div>

<script>
// ── Drag & drop visual highlight for file uploader dropzones ─────────────
function attachDragListeners() {
    document.querySelectorAll('div[data-testid="stFileUploaderDropzone"]').forEach(zone => {
        if (zone.dataset.dragBound) return;
        zone.dataset.dragBound = "1";
        ["dragenter", "dragover"].forEach(evt =>
            zone.addEventListener(evt, e => {
                e.preventDefault();
                zone.classList.add("dragover");
            })
        );
        ["dragleave", "drop"].forEach(evt =>
            zone.addEventListener(evt, () => zone.classList.remove("dragover"))
        );
    });
}
attachDragListeners();
const dragObserver = new MutationObserver(attachDragListeners);
dragObserver.observe(document.body, { childList: true, subtree: true });

// ── Keyboard shortcut: Enter triggers the primary Analyze button ─────────
document.addEventListener("keydown", function(e) {
    if (e.key !== "Enter") return;
    const active = document.activeElement;
    if (active && (active.tagName === "TEXTAREA" || active.tagName === "INPUT")) return;
    const btn = document.querySelector('button[kind="primary"]');
    if (btn) {
        e.preventDefault();
        btn.click();
    }
});
</script>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Audio chime (loaded from assets/chime.wav as base64, with safe fallback)
# ─────────────────────────────────────────────────────────────────────────────
import base64 as _base64

@st.cache_data(show_spinner=False)
def _load_chime_data_uri():
    try:
        chime_path = Path(__file__).parent / "assets" / "chime.wav"
        with open(chime_path, "rb") as f:
            b64 = _base64.b64encode(f.read()).decode()
        return f"data:audio/wav;base64,{b64}"
    except Exception:
        return None

CHIME_DATA_URI = _load_chime_data_uri()



# ─────────────────────────────────────────────────────────────────────────────
# Cached Resources
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading face detector...")
def load_face_detector():
    from mtcnn import MTCNN
    return MTCNN()


@st.cache_resource(show_spinner="Loading DeepGuard AI model...")
def load_model():
    import tensorflow as tf

    custom_objects = {**CUSTOM_OBJECTS, "tf": tf}

    candidates = []
    for ext_path in [
        MODEL_PATH,
        MODELS_DIR / "deepfake_detector.h5",
        MODEL_PATH_H5,
        MODELS_DIR / "deepfake_detector_phase1.h5",
    ]:
        if ext_path.exists():
            candidates.append(ext_path)

    if not candidates:
        return None

    for target in candidates:
        try:
            model = tf.keras.models.load_model(
                str(target),
                custom_objects=custom_objects,
                compile=False,
            )
            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            return model
        except Exception:
            try:
                model = build_model(trainable_base=False)
                model.load_weights(str(target))
                return model
            except Exception:
                continue

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Inference Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _crop_face(frame_bgr: np.ndarray, bbox: tuple):
    x, y, w, h = bbox
    pad_x = int(w * FACE_PADDING)
    pad_y = int(h * FACE_PADDING)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_bgr.shape[1], x + w + pad_x)
    y2 = min(frame_bgr.shape[0], y + h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]


def _calculate_confidence(score: float, threshold: float = None) -> float:
    if threshold is None:
        threshold = DECISION_THRESHOLD
    if score >= threshold:
        conf = (score - threshold) / (1.0 - threshold)
    else:
        conf = (threshold - score) / threshold
    return max(0.0, min(1.0, conf))


def analyze_image(image_data, model, detector):
    if isinstance(image_data, Image.Image):
        img_rgb = np.array(image_data.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image_data

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(img_rgb)

    if detections:
        if len(detections) == 1:
            x, y, w, h = detections[0]["box"]
            img_h, img_w = img_rgb.shape[:2]
            face_area = w * h
            img_area = img_w * img_h
            if face_area / img_area >= 0.65 or (img_w <= 256 and img_h <= 256):
                resized = cv2.resize(img_rgb, IMG_SIZE)
                faces_array = np.array([resized], dtype=np.float32)
                predictions = model.predict(faces_array, batch_size=1, verbose=0)
                avg_score = float(predictions[0][0])
                return avg_score, 1, faces_array

        faces = []
        for det in detections:
            cropped = _crop_face(img_bgr, det["box"])
            if cropped is not None:
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(cropped_rgb, IMG_SIZE)
                faces.append(resized)

        if faces:
            faces_array = np.array(faces, dtype=np.float32)
            predictions = model.predict(faces_array, batch_size=BATCH_SIZE, verbose=0)
            avg_score = float(np.mean(predictions.flatten()))
            return avg_score, len(faces), faces_array

    resized = cv2.resize(img_rgb, IMG_SIZE)
    full_img = np.array([resized], dtype=np.float32)
    predictions = model.predict(full_img, batch_size=1, verbose=0)
    avg_score = float(predictions[0][0])
    return avg_score, 0, full_img


def analyze_video(video_path, model, detector, progress_callback=None, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None, [], 0, 0, 0

    faces_collected = []
    missed_count = 0
    frames_sampled = 0
    current_frame = 0

    while len(faces_collected) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % FRAME_STEP == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected = detector.detect_faces(rgb_frame)
            frames_sampled += 1

            if detected:
                best = max(detected, key=lambda d: d["confidence"])
                cropped = _crop_face(frame, best["box"])
                if cropped is not None:
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(cropped_rgb, IMG_SIZE)
                    faces_collected.append(resized)
                else:
                    missed_count += 1
            else:
                missed_count += 1

            if progress_callback:
                progress_callback(frames_sampled / (total_frames / FRAME_STEP))

        current_frame += 1

    cap.release()

    if not faces_collected:
        return None, [], 0, missed_count, frames_sampled

    faces_array = np.array(faces_collected, dtype=np.float32)
    predictions = model.predict(faces_array, batch_size=BATCH_SIZE, verbose=0)
    per_frame_scores = predictions.flatten().tolist()
    avg_score = float(np.mean(per_frame_scores))

    return avg_score, per_frame_scores, len(faces_collected), missed_count, frames_sampled


# ─────────────────────────────────────────────────────────────────────────────
# UI Render Helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_score_bar(score, is_fake):
    pct = int(score * 100)
    fill_class = "score-bar-fill-fake" if is_fake else "score-bar-fill-real"
    st.markdown(f"""
    <div class="score-container">
        <div class="score-bar-bg">
            <div class="{fill_class}" style="width: {pct}%;"></div>
        </div>
        <div class="score-label">
            <span>AI Generated (Fake)</span>
            <span>{score:.3f}</span>
            <span>Authentic (Real)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_confidence_ring(confidence, is_fake):
    """SVG animated circular confidence gauge."""
    pct = confidence * 100
    radius = 54
    circumference = 2 * 3.14159265 * radius
    offset = circumference * (1 - confidence)
    color1, color2 = ("#ff416c", "#ff4b2b") if is_fake else ("#11998e", "#38ef7d")

    svg = f"""
    <div class="ring-wrap">
    <svg width="140" height="140" viewBox="0 0 140 140">
        <defs>
            <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="{color1}" />
                <stop offset="100%" stop-color="{color2}" />
            </linearGradient>
        </defs>
        <circle cx="70" cy="70" r="{radius}" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="12"/>
        <circle cx="70" cy="70" r="{radius}" fill="none" stroke="url(#ringGrad)" stroke-width="12"
            stroke-linecap="round"
            stroke-dasharray="{circumference:.2f}"
            stroke-dashoffset="{circumference:.2f}"
            transform="rotate(-90 70 70)">
            <animate attributeName="stroke-dashoffset" from="{circumference:.2f}" to="{offset:.2f}" dur="1.2s" fill="freeze" calcMode="spline" keySplines="0.2 0.8 0.2 1"/>
        </circle>
        <text x="70" y="64" text-anchor="middle" fill="#e0e0ff" font-size="26" font-weight="800" font-family="Inter">{pct:.0f}%</text>
        <text x="70" y="86" text-anchor="middle" fill="#7a7a9e" font-size="11" font-family="Inter" letter-spacing="1">CONFIDENCE</text>
    </svg>
    </div>
    """
    st.markdown(svg, unsafe_allow_html=True)


def render_verdict(is_fake, confidence, score, inference_time=None):
    if is_fake:
        st.markdown('<div class="verdict-fake">🚨 AI-GENERATED CONTENT DETECTED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict-real">✅ AUTHENTIC CONTENT VERIFIED</div>', unsafe_allow_html=True)

    if inference_time is not None:
        st.markdown(
            f'<div style="text-align:center; margin-top:0.6rem;">'
            f'<span class="chip chip-neutral">⏱️ Processed in {inference_time:.2f}s</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    render_score_bar(score, is_fake)


def render_metrics(confidence, face_count, extra_metrics=None):
    cols = st.columns(3 if extra_metrics else 2)

    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{confidence*100:.1f}%</div>
            <div class="metric-sub">Model certainty in verdict</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        face_text = f"{face_count}" if face_count > 0 else "Full Image"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Faces Analyzed</div>
            <div class="metric-value">{face_text}</div>
            <div class="metric-sub">{"MTCNN face detection" if face_count > 0 else "No faces found, analyzed full image"}</div>
        </div>
        """, unsafe_allow_html=True)

    if extra_metrics:
        with cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{extra_metrics["label"]}</div>
                <div class="metric-value">{extra_metrics["value"]}</div>
                <div class="metric-sub">{extra_metrics["sub"]}</div>
            </div>
            """, unsafe_allow_html=True)


def generate_result_card_image(filename, kind, is_fake, confidence, score):
    """Generates a styled PNG result card (PIL) for sharing/portfolio screenshots."""
    W, H = 900, 500
    bg_top = (10, 10, 26)
    bg_bottom = (15, 17, 23)

    img = Image.new("RGB", (W, H), bg_top)
    draw = ImageDraw.Draw(img)

    # Vertical gradient background
    for y in range(H):
        ratio = y / H
        r = int(bg_top[0] + (bg_bottom[0] - bg_top[0]) * ratio)
        g = int(bg_top[1] + (bg_bottom[1] - bg_top[1]) * ratio)
        b = int(bg_top[2] + (bg_bottom[2] - bg_top[2]) * ratio)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    accent = (255, 65, 108) if is_fake else (56, 239, 125)
    accent2 = (255, 75, 43) if is_fake else (17, 153, 142)

    # Verdict banner
    banner_h = 110
    for x in range(W):
        ratio = x / W
        r = int(accent[0] + (accent2[0] - accent[0]) * ratio)
        g = int(accent[1] + (accent2[1] - accent[1]) * ratio)
        b = int(accent[2] + (accent2[2] - accent[2]) * ratio)
        draw.line([(x, 40), (x, 40 + banner_h)], fill=(r, g, b))

    def _font(size, bold=True):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for c in candidates:
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
        return ImageFont.load_default()

    title_font = _font(34)
    label_font = _font(20)
    value_font = _font(26)
    small_font = _font(16, bold=False)
    brand_font = _font(22)

    verdict_text = "AI-GENERATED CONTENT DETECTED" if is_fake else "AUTHENTIC CONTENT VERIFIED"
    icon = "🚨" if is_fake else "✅"

    # Center verdict text on banner
    bbox = draw.textbbox((0, 0), verdict_text, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, 40 + (banner_h - 34) // 2 - 4), verdict_text, font=title_font, fill=(255, 255, 255))

    # Score bar
    bar_y = 200
    bar_w = W - 120
    bar_x = 60
    draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + 18], radius=9, fill=(40, 40, 60))
    fill_w = int(bar_w * score)
    draw.rounded_rectangle([bar_x, bar_y, bar_x + max(fill_w, 18), bar_y + 18], radius=9, fill=accent)

    draw.text((bar_x, bar_y + 28), "AI Generated (Fake)", font=small_font, fill=(122, 122, 158))
    score_label = f"{score:.3f}"
    bbox = draw.textbbox((0, 0), score_label, font=small_font)
    draw.text(((W - (bbox[2] - bbox[0])) // 2, bar_y + 28), score_label, font=small_font, fill=(122, 122, 158))
    bbox = draw.textbbox((0, 0), "Authentic (Real)", font=small_font)
    draw.text((bar_x + bar_w - (bbox[2] - bbox[0]), bar_y + 28), "Authentic (Real)", font=small_font, fill=(122, 122, 158))

    # Metric cards
    card_y = 280
    card_h = 110
    card_w = (W - 120 - 2 * 20) // 3
    metrics = [
        ("CONFIDENCE", f"{confidence*100:.1f}%"),
        ("FILE", filename[:18] + ("…" if len(filename) > 18 else "")),
        ("TYPE", kind),
    ]
    for i, (label, value) in enumerate(metrics):
        cx = 60 + i * (card_w + 20)
        draw.rounded_rectangle([cx, card_y, cx + card_w, card_y + card_h], radius=14, fill=(28, 28, 42), outline=(50, 50, 70))
        draw.text((cx + 18, card_y + 18), label, font=small_font, fill=(122, 122, 158))
        draw.text((cx + 18, card_y + 45), str(value), font=value_font, fill=(224, 224, 255))

    # Footer / branding
    footer_y = H - 70
    draw.line([(60, footer_y - 15), (W - 60, footer_y - 15)], fill=(40, 40, 60))
    draw.text((60, footer_y), "🛡️ DeepGuard AI", font=brand_font, fill=(108, 99, 255))
    threshold_text = f"Threshold: {DECISION_THRESHOLD}  •  EfficientNetV2 + CBAM"
    bbox = draw.textbbox((0, 0), threshold_text, font=small_font)
    draw.text((W - 60 - (bbox[2] - bbox[0]), footer_y + 5), threshold_text, font=small_font, fill=(90, 90, 126))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def render_result_actions(filename, kind, is_fake, confidence, score):
    """Renders an interactive 'copy result summary' button with clipboard feedback."""
    verdict_text = "AI-GENERATED (DEEPFAKE)" if is_fake else "AUTHENTIC"
    summary = (
        f"DeepGuard AI Report\\n"
        f"File: {filename}\\n"
        f"Type: {kind}\\n"
        f"Verdict: {verdict_text}\\n"
        f"Confidence: {confidence*100:.1f}%\\n"
        f"Raw Score: {score:.3f}\\n"
        f"Threshold: {DECISION_THRESHOLD}"
    ).replace('"', '\\"')

    accent = "#ff416c" if is_fake else "#38ef7d"

    components.html(f"""
    <div style="display:flex; justify-content:flex-end; margin-top:0.6rem;">
        <button id="copyBtn" onclick="copyResult()" style="
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.1);
            color: #b0b0d0;
            border-radius: 10px;
            padding: 0.5rem 1.1rem;
            font-size: 0.82rem;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            transition: all 0.25s cubic-bezier(0.165, 0.84, 0.44, 1);
        " onmouseover="this.style.borderColor='{accent}'; this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 18px {accent}33';"
           onmouseout="this.style.borderColor='rgba(255,255,255,0.1)'; this.style.transform='translateY(0)'; this.style.boxShadow='none';">
            <span id="copyIcon">📋</span> <span id="copyLabel">Copy Result Summary</span>
        </button>
    </div>
    <script>
        function copyResult() {{
            const text = "{summary}";
            navigator.clipboard.writeText(text).then(() => {{
                document.getElementById("copyIcon").innerText = "✅";
                document.getElementById("copyLabel").innerText = "Copied!";
                setTimeout(() => {{
                    document.getElementById("copyIcon").innerText = "📋";
                    document.getElementById("copyLabel").innerText = "Copy Result Summary";
                }}, 1800);
            }});
        }}
    </script>
    """, height=50)

    card_buf = generate_result_card_image(filename, kind, is_fake, confidence, score)
    dl_col1, dl_col2 = st.columns([3, 1])
    with dl_col2:
        st.download_button(
            "🖼️ Download Result Card",
            data=card_buf,
            file_name="deepguard_result.png",
            mime="image/png",
            use_container_width=True,
            key=f"download_card_{filename}_{score:.4f}",
        )


def add_history(filename, kind, is_fake, confidence, score):
    st.session_state.history.insert(0, {
        "name": filename,
        "kind": kind,
        "verdict": "FAKE" if is_fake else "REAL",
        "confidence": confidence,
        "score": score,
        "time": time.strftime("%H:%M:%S"),
    })
    st.session_state.history = st.session_state.history[:8]
    st.session_state.scan_count += 1

    # Track "authentic streak"
    if not is_fake:
        st.session_state.real_streak = st.session_state.get("real_streak", 0) + 1
    else:
        st.session_state.real_streak = 0


def render_typing_subtitle():
    """Renders the hero subtitle with a one-time typewriter animation."""
    text = (
        "Upload images or videos to instantly detect AI-generated deepfakes. "
        "Powered by EfficientNetV2 + CBAM attention — the same model used in the Chrome extension."
    )
    components.html(f"""
    <style>
        body {{ margin: 0; background: transparent; }}
        .typing-p {{
            font-family: 'Inter', sans-serif;
            color: #7a7a9e;
            font-size: 1.05rem;
            max-width: 620px;
            margin: 0 auto;
            line-height: 1.6;
            text-align: center;
            min-height: 3.2em;
        }}
        .typing-cursor {{
            display: inline-block;
            width: 2px;
            background: #6C63FF;
            margin-left: 2px;
            animation: blinkCursor 0.8s step-end infinite;
        }}
        @keyframes blinkCursor {{ 50% {{ opacity: 0; }} }}
    </style>
    <p class="typing-p"><span id="typed"></span><span class="typing-cursor">&nbsp;</span></p>
    <script>
        const text = {text!r};
        const el = document.getElementById("typed");
        let i = 0;
        function type() {{
            if (i <= text.length) {{
                el.textContent = text.slice(0, i);
                i++;
                setTimeout(type, 14);
            }}
        }}
        type();
    </script>
    """, height=100)


def render_stats_strip():
    """Renders the hero stats strip with animated count-up numbers (self-contained component)."""
    components.html("""
    <style>
        body { margin: 0; font-family: 'Inter', sans-serif; background: transparent; }
        .stats-strip {
            display: flex;
            justify-content: center;
            gap: 1.2rem;
            flex-wrap: wrap;
            padding: 0.5rem 0;
        }
        .stat-pill {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 100px;
            padding: 0.5rem 1.3rem;
            font-size: 0.82rem;
            color: #b0b0d0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.25s ease;
        }
        .stat-pill:hover {
            border-color: rgba(108,99,255,0.4);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(108,99,255,0.15);
        }
        .stat-pill b { color: #e0e0ff; font-weight: 800; }
    </style>
    <div class="stats-strip">
        <div class="stat-pill">🎯 <b><span class="counter" data-target="95" data-suffix="%">0%</span></b>&nbsp;Accuracy</div>
        <div class="stat-pill">⚡ <b>Real-time</b>&nbsp;Inference</div>
        <div class="stat-pill">🔒 <b><span class="counter" data-target="100" data-suffix="%">0%</span></b>&nbsp;Local & Private</div>
        <div class="stat-pill">🧠 <b><span class="counter" data-target="140" data-suffix="K">0K</span></b>&nbsp;Training Images</div>
    </div>
    <script>
        document.querySelectorAll(".counter").forEach(el => {
            const target = el.dataset.target;
            const suffix = el.dataset.suffix || "";
            const end = parseFloat(target);
            const duration = 1200;
            const start = performance.now();
            function tick(now) {
                const progress = Math.min((now - start) / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);
                const value = end * eased;
                el.textContent = (end % 1 === 0 ? Math.round(value) : value.toFixed(1)) + suffix;
                if (progress < 1) requestAnimationFrame(tick);
            }
            requestAnimationFrame(tick);
        });
    </script>
    """, height=60)


def render_sparkle_burst():
    """Renders a brief sparkle/particle burst (used on REAL verdicts only)."""
    sparkles_html = ""
    colors = ["#38ef7d", "#48CAE4", "#6C63FF", "#ffd166"]
    positions = [
        (-120, -40), (-70, -90), (-20, -60), (30, -95),
        (80, -45), (110, -75), (-100, -10), (100, -15),
    ]
    for i, (dx, dy) in enumerate(positions):
        color = colors[i % len(colors)]
        sparkles_html += (
            f'<div class="sparkle" style="--dx:{dx}px; --dy:{dy}px; '
            f'background:{color}; box-shadow:0 0 8px {color}, 0 0 16px {color}66; '
            f'animation-delay:{i * 0.04}s;"></div>'
        )
    st.markdown(f'<div class="sparkle-burst">{sparkles_html}</div>', unsafe_allow_html=True)


def render_chime():
    """Plays a short audio chime if sound is enabled and the asset is available."""
    if not st.session_state.get("sound_enabled", False):
        return
    if not CHIME_DATA_URI:
        return
    components.html(f"""
    <audio autoplay style="display:none;">
        <source src="{CHIME_DATA_URI}" type="audio/wav">
    </audio>
    """, height=0)


def render_history_sidebar():
    with st.sidebar:
        st.markdown("### 📜 Scan History")

        if not st.session_state.history:
            st.caption("No scans yet — results will appear here.")
        else:
            filter_choice = st.radio(
                "Filter",
                ["All", "Real", "Fake"],
                horizontal=True,
                label_visibility="collapsed",
                key="history_filter",
            )

            if filter_choice == "Real":
                filtered = [h for h in st.session_state.history if h["verdict"] == "REAL"]
            elif filter_choice == "Fake":
                filtered = [h for h in st.session_state.history if h["verdict"] == "FAKE"]
            else:
                filtered = st.session_state.history

            if not filtered:
                st.caption(f"No {filter_choice.lower()} results in history yet.")

            for idx, item in enumerate(filtered):
                chip = "chip-fake" if item["verdict"] == "FAKE" else "chip-real"
                st.markdown(f"""
                <div class="history-row">
                    <div>
                        <span class="chip {chip}">{item['verdict']}</span><br>
                        <span style="font-size:0.75rem; color:#7a7a9e;">{item['name'][:22]}</span>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-weight:700; color:#e0e0ff;">{item['confidence']*100:.0f}%</div>
                        <div style="font-size:0.7rem; color:#5a5a7e;">{item['time']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("Details", expanded=False):
                    st.caption(f"**Type:** {item['kind']}")
                    st.caption(f"**Raw Score:** {item['score']:.3f}")
                    st.caption(f"**Confidence:** {item['confidence']*100:.1f}%")
                    st.caption(f"**Threshold:** {DECISION_THRESHOLD}")
                    st.caption(f"**Scanned at:** {item['time']}")

        st.markdown("---")
        st.markdown(f"""
        <div class="glass-card-sm" style="text-align:center;">
            <div class="metric-label">Total Scans This Session</div>
            <div class="metric-value">{st.session_state.scan_count}</div>
        </div>
        """, unsafe_allow_html=True)

        streak = st.session_state.get("real_streak", 0)
        if streak >= 2:
            st.markdown(
                f'<div style="text-align:center; margin-top:0.6rem;">'
                f'<span class="streak-badge">🔥 {streak} authentic in a row</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("##### ⚙️ Detection Settings")
        st.caption(f"Decision Threshold: **{DECISION_THRESHOLD}**")
        st.caption("Model: EfficientNetV2-B0 + CBAM")
        st.caption("Face Detector: MTCNN")

        st.markdown("---")
        st.toggle("🔔 Sound on verdict", key="sound_enabled", value=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Hero Header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-badge"><span class="dot"></span>AI-Powered Detection Engine — Live</div>
        <h1>DeepGuard AI</h1>
    </div>
    """, unsafe_allow_html=True)

    render_typing_subtitle()
    render_stats_strip()

    st.markdown("""
    <div class="flip-grid">
        <div class="flip-card">
            <div class="flip-inner">
                <div class="flip-face flip-front">
                    <div class="flip-icon">🧬</div>
                    <div class="flip-title">Face Extraction</div>
                </div>
                <div class="flip-face flip-back">
                    <div class="flip-title">MTCNN Detection</div>
                    <div class="flip-desc">Locates and crops every face in your image or video frames with precision padding.</div>
                </div>
            </div>
        </div>
        <div class="flip-card">
            <div class="flip-inner">
                <div class="flip-face flip-front">
                    <div class="flip-icon">🧠</div>
                    <div class="flip-title">Deep Analysis</div>
                </div>
                <div class="flip-face flip-back">
                    <div class="flip-title">EfficientNetV2 + CBAM</div>
                    <div class="flip-desc">Runs each face through a CNN with channel & spatial attention to spot synthetic artifacts.</div>
                </div>
            </div>
        </div>
        <div class="flip-card">
            <div class="flip-inner">
                <div class="flip-face flip-front">
                    <div class="flip-icon">📊</div>
                    <div class="flip-title">Confidence Score</div>
                </div>
                <div class="flip-face flip-back">
                    <div class="flip-title">Normalized Certainty</div>
                    <div class="flip-desc">Scores are calibrated against a decision threshold to produce a clear confidence percentage.</div>
                </div>
            </div>
        </div>
        <div class="flip-card">
            <div class="flip-inner">
                <div class="flip-face flip-front">
                    <div class="flip-icon">🔒</div>
                    <div class="flip-title">Private by Design</div>
                </div>
                <div class="flip-face flip-back">
                    <div class="flip-title">Runs Locally</div>
                    <div class="flip-desc">No images or videos ever leave your machine — all inference happens on-device.</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Check for Keras 3 environment mismatch
    is_keras_3 = False
    try:
        import keras
        if keras.__version__.startswith("3"):
            is_keras_3 = True
    except Exception:
        import tensorflow as tf
        if hasattr(tf, "keras") and hasattr(tf.keras, "__version__") and tf.keras.__version__.startswith("3"):
            is_keras_3 = True

    if is_keras_3:
        st.error("""
        ### ⚠️ Compatibility Warning: Keras 3 Environment Detected

        DeepGuard AI model weights were trained using **Keras 2 (TensorFlow 2.13 - 2.15)**.
        Running under Keras 3 (TensorFlow 2.16+) causes layers to align incorrectly,
        resulting in incorrect predictions (e.g. AI-generated images incorrectly classified as REAL).

        **To fix this, please run the application in the `deepguard` conda environment:**

        1. **Stop** this current streamlit server (press `Ctrl+C` in your terminal).
        2. **Activate** the deepguard conda environment:
           ```bash
           conda activate deepguard
           ```
        3. **Relaunch** the application:
           ```bash
           streamlit run app.py
           ```
        """)

    # ── Load Model ───────────────────────────────────────────────────────────
    model = load_model()
    detector = load_face_detector()

    if model is None:
        st.error("No trained model found. Please train the model first.")
        st.markdown("""
        <div class="glass-card">
            <h4>How to train:</h4>
            <ol>
                <li>Download dataset: <code>python download_data.py</code></li>
                <li>Preprocess data: <code>python src/preprocess.py</code></li>
                <li>Train model: <code>python src/train.py</code></li>
                <li>Reload this page</li>
            </ol>
            <p>Or use the Colab notebook at <code>notebooks/DeepGuard_AI_Training.ipynb</code></p>
        </div>
        """, unsafe_allow_html=True)
        return

    render_history_sidebar()

    # ── Mode Selection via Tabs ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    tab_img, tab_vid, tab_live = st.tabs(["🖼️  Image Analysis", "🎬  Video Analysis", "📷  Live Capture"])

    # ══════════════════════════════════════════════════════════════════════════
    # IMAGE MODE
    # ══════════════════════════════════════════════════════════════════════════
    with tab_img:
        col_upload, col_result = st.columns([1, 1], gap="large")

        with col_upload:
            uploaded_image = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                label_visibility="collapsed",
                key=f"img_uploader_{st.session_state.img_uploader_version}",
            )

            if uploaded_image:
                image = Image.open(uploaded_image)
                st.markdown('<div class="glass-card slide-up" style="padding:1rem;">', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.8rem; flex-wrap:wrap; gap:0.4rem;">
                        <div>
                            <span class="chip chip-neutral">FILE</span>
                            <span style="color:#b0b0d0; font-size:0.85rem;">{uploaded_image.name}</span>
                        </div>
                        <div style="color:#5a5a7e; font-size:0.8rem;">
                            {uploaded_image.size / 1024:.1f} KB &nbsp;•&nbsp; {image.size[0]} × {image.size[1]} px
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="glass-card slide-up" style="text-align:center;">
                    <div class="upload-icon">📸</div>
                    <div class="upload-title">Upload Image for Analysis</div>
                    <div class="upload-sub">Supports: JPG, PNG, WEBP, BMP (max 200MB)</div>
                </div>
                """, unsafe_allow_html=True)

        with col_result:
            if uploaded_image:
                analyze_btn = st.button(
                    "🔍 Analyze Image",
                    type="primary",
                    use_container_width=True,
                    key="analyze_img",
                )

                if analyze_btn:
                    loader_placeholder = st.empty()
                    with loader_placeholder.container():
                        st.markdown(
                            "<h5 style='text-align:center; color:#8b8bff;'>Analyzing Spatial Pixel Frequency...</h5>",
                            unsafe_allow_html=True,
                        )
                        if LOTTIE_PROCESSING:
                            st_lottie(LOTTIE_PROCESSING, height=180, key="img_processing_loader")
                        else:
                            st.progress(60, text="Running EfficientNetV2 + CBAM inference...")

                    _t0 = time.perf_counter()
                    score, face_count, faces = analyze_image(image, model, detector)
                    inference_time = time.perf_counter() - _t0
                    loader_placeholder.empty()

                    is_fake = score < DECISION_THRESHOLD
                    confidence = _calculate_confidence(score)

                    render_verdict(is_fake, confidence, score, inference_time=inference_time)
                    if not is_fake:
                        render_sparkle_burst()
                    render_chime()
                    st.markdown("<br>", unsafe_allow_html=True)

                    ring_col, met_col = st.columns([1, 2])
                    with ring_col:
                        render_confidence_ring(confidence, is_fake)
                    with met_col:
                        render_metrics(confidence, face_count)

                    render_result_actions(uploaded_image.name, "Image", is_fake, confidence, score)

                    add_history(uploaded_image.name, "Image", is_fake, confidence, score)

                    if confidence < 0.40:
                        st.warning(
                            "⚠️ **Low Confidence Prediction.** The model is not very certain "
                            "about this result. This can happen with sunglasses, heavy makeup, "
                            "unusual lighting, or non-standard face angles."
                        )

                    if face_count == 0:
                        st.warning(
                            "⚠️ **No faces detected.** The model analyzed the full image. "
                            "Since the model was trained specifically on cropped faces, "
                            "this prediction may be inaccurate. For best results, please upload "
                            "an image containing a clear, front-facing face or crop the face closely before uploading."
                        )

                    if is_fake:
                        st.toast("🚨 Deepfake detected!", icon="🚨")
                    else:
                        st.toast("✅ Looks authentic!", icon="✅")

                    if face_count > 0 and faces is not None:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("##### 🧬 Detected Faces")
                        face_cols = st.columns(min(face_count, 5))
                        for i, face in enumerate(faces):
                            with face_cols[i % 5]:
                                st.image(face.astype(np.uint8), caption=f"Face {i+1}", width=120)

                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🔄 Scan Another Image", use_container_width=True, key="reset_img"):
                        st.session_state.img_uploader_version += 1
                        st.rerun()
            else:
                st.markdown('<div class="glass-card fade-in" style="text-align:center; padding: 2.5rem 2rem;">', unsafe_allow_html=True)
                if LOTTIE_SHIELD:
                    st_lottie(LOTTIE_SHIELD, height=220, key="img_welcome_shield")
                else:
                    st.markdown("<div style='font-size: 4rem; margin-bottom: 1rem;'>🛡️</div>", unsafe_allow_html=True)
                st.markdown("""
                    <div style="color: #7a7a9e; font-size: 1.1rem; font-weight:600; margin-top:0.5rem;">
                        Awaiting Forensic Core Initialization
                    </div>
                    <div style="color: #4a4a6e; font-size: 0.85rem; margin-top: 0.5rem;">
                        Upload an image to begin analysis &mdash; the model detects AI-generated faces with 90-95% accuracy
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # VIDEO MODE
    # ══════════════════════════════════════════════════════════════════════════
    with tab_vid:
        col_upload, col_result = st.columns([1, 1], gap="large")

        with col_upload:
            uploaded_video = st.file_uploader(
                "Choose a video",
                type=["mp4", "mov", "avi", "mkv"],
                label_visibility="collapsed",
                key=f"vid_uploader_{st.session_state.vid_uploader_version}",
            )

            if uploaded_video:
                st.markdown('<div class="glass-card slide-up" style="padding:1rem;">', unsafe_allow_html=True)
                st.video(uploaded_video)
                st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.8rem; flex-wrap:wrap; gap:0.4rem;">
                        <div>
                            <span class="chip chip-neutral">FILE</span>
                            <span style="color:#b0b0d0; font-size:0.85rem;">{uploaded_video.name}</span>
                        </div>
                        <div style="color:#5a5a7e; font-size:0.8rem;">
                            {uploaded_video.size / (1024*1024):.1f} MB
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="glass-card slide-up" style="text-align:center;">
                    <div class="upload-icon">🎬</div>
                    <div class="upload-title">Upload Video for Analysis</div>
                    <div class="upload-sub">Supports: MP4, MOV, AVI, MKV (max 200MB)</div>
                </div>
                """, unsafe_allow_html=True)

        with col_result:
            if uploaded_video:
                analyze_btn = st.button(
                    "🔍 Analyze Video",
                    type="primary",
                    use_container_width=True,
                    key="analyze_vid",
                )

                if analyze_btn:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    video_path = tfile.name
                    tfile.write(uploaded_video.read())
                    tfile.close()

                    try:
                        lottie_col, bar_col = st.columns([1, 3])
                        with lottie_col:
                            if LOTTIE_PROCESSING:
                                st_lottie(LOTTIE_PROCESSING, height=90, key="vid_processing_loader")
                        with bar_col:
                            st.markdown(
                                "<div style='padding-top:1.2rem; color:#8b8bff; font-weight:600;'>Analyzing Spatial Pixel Frequency...</div>",
                                unsafe_allow_html=True,
                            )
                        progress_bar = st.progress(0, text="Detecting faces and analyzing frames...")

                        def update_progress(pct):
                            progress_bar.progress(min(pct, 1.0), text=f"Analyzing frames... {int(min(pct,1.0)*100)}%")

                        _t0 = time.perf_counter()
                        score, per_frame, face_count, missed, total_sampled = analyze_video(
                            video_path, model, detector,
                            progress_callback=update_progress,
                            max_frames=100,
                        )
                        inference_time = time.perf_counter() - _t0

                        progress_bar.empty()

                        if score is None:
                            st.error("Could not extract any faces from the video.")
                        else:
                            is_fake = score < DECISION_THRESHOLD
                            confidence = _calculate_confidence(score)

                            render_verdict(is_fake, confidence, score, inference_time=inference_time)
                            if not is_fake:
                                render_sparkle_burst()
                            render_chime()
                            st.markdown("<br>", unsafe_allow_html=True)

                            fake_frame_pct = float(np.mean(np.array(per_frame) < DECISION_THRESHOLD)) * 100

                            ring_col, met_col = st.columns([1, 2])
                            with ring_col:
                                render_confidence_ring(confidence, is_fake)
                            with met_col:
                                render_metrics(confidence, face_count, extra_metrics={
                                    "label": "Fake Frames",
                                    "value": f"{fake_frame_pct:.0f}%",
                                    "sub": f"{face_count} frames analyzed",
                                })

                            render_result_actions(uploaded_video.name, "Video", is_fake, confidence, score)

                            add_history(uploaded_video.name, "Video", is_fake, confidence, score)

                            if missed > 0:
                                pct = missed / max(total_sampled, 1) * 100
                                st.markdown(
                                    f'<div class="warning-box">No face detected in '
                                    f'{missed} of {total_sampled} sampled frames '
                                    f'({pct:.0f}%). These frames were safely skipped.</div>',
                                    unsafe_allow_html=True,
                                )

                            if is_fake:
                                st.toast("🚨 Deepfake detected in video!", icon="🚨")
                            else:
                                st.toast("✅ Video looks authentic!", icon="✅")

                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("##### 📈 Per-Frame Analysis")
                            import pandas as pd
                            chart_data = pd.DataFrame({
                                "Model Score": per_frame,
                                "Threshold": [DECISION_THRESHOLD] * len(per_frame),
                            })
                            st.line_chart(chart_data, use_container_width=True)

                            st.markdown("<br>", unsafe_allow_html=True)
                            with st.expander("🔬 Frame-by-Frame Breakdown"):
                                fake_n = int(np.sum(np.array(per_frame) < DECISION_THRESHOLD))
                                real_n = len(per_frame) - fake_n
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Fake-Leaning Frames</div>
                                        <div class="metric-value" style="color:#ff7a93;">{fake_n}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                with c2:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Real-Leaning Frames</div>
                                        <div class="metric-value" style="color:#6dffb0;">{real_n}</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button("🔄 Scan Another Video", use_container_width=True, key="reset_vid"):
                                st.session_state.vid_uploader_version += 1
                                st.rerun()

                    finally:
                        try:
                            os.remove(video_path)
                        except OSError:
                            pass
            else:
                st.markdown('<div class="glass-card fade-in" style="text-align:center; padding: 2.5rem 2rem;">', unsafe_allow_html=True)
                if LOTTIE_SHIELD:
                    st_lottie(LOTTIE_SHIELD, height=220, key="vid_welcome_shield")
                else:
                    st.markdown("<div style='font-size: 4rem; margin-bottom: 1rem;'>🛡️</div>", unsafe_allow_html=True)
                st.markdown("""
                    <div style="color: #7a7a9e; font-size: 1.1rem; font-weight:600; margin-top:0.5rem;">
                        Awaiting Forensic Core Initialization
                    </div>
                    <div style="color: #4a4a6e; font-size: 0.85rem; margin-top: 0.5rem;">
                        Upload a video to begin analysis &mdash; DeepGuard AI scans each frame for AI-generated content
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE CAPTURE MODE
    # ══════════════════════════════════════════════════════════════════════════
    with tab_live:
        col_upload, col_result = st.columns([1, 1], gap="large")

        with col_upload:
            st.markdown("""
            <div class="glass-card slide-up" style="text-align:center; margin-bottom:1rem;">
                <div class="upload-icon">📷</div>
                <div class="upload-title">Live Webcam Capture</div>
                <div class="upload-sub">Take a photo for instant deepfake analysis</div>
            </div>
            """, unsafe_allow_html=True)

            captured_image = st.camera_input(
                "Take a photo",
                label_visibility="collapsed",
                key=f"cam_uploader_{st.session_state.cam_uploader_version}",
            )

        with col_result:
            if captured_image:
                live_image = Image.open(captured_image)
                analyze_btn = st.button(
                    "🔍 Analyze Capture",
                    type="primary",
                    use_container_width=True,
                    key="analyze_live",
                )

                if analyze_btn:
                    loader_placeholder = st.empty()
                    with loader_placeholder.container():
                        st.markdown(
                            "<h5 style='text-align:center; color:#8b8bff;'>Analyzing Spatial Pixel Frequency...</h5>",
                            unsafe_allow_html=True,
                        )
                        if LOTTIE_PROCESSING:
                            st_lottie(LOTTIE_PROCESSING, height=180, key="live_processing_loader")
                        else:
                            st.progress(60, text="Running EfficientNetV2 + CBAM inference...")

                    _t0 = time.perf_counter()
                    score, face_count, faces = analyze_image(live_image, model, detector)
                    inference_time = time.perf_counter() - _t0
                    loader_placeholder.empty()

                    # Flip the score for live webcam capture as requested
                    score = 1.0 - score

                    is_fake = score < DECISION_THRESHOLD
                    confidence = _calculate_confidence(score)

                    render_verdict(is_fake, confidence, score, inference_time=inference_time)
                    if not is_fake:
                        render_sparkle_burst()
                    render_chime()
                    st.markdown("<br>", unsafe_allow_html=True)

                    ring_col, met_col = st.columns([1, 2])
                    with ring_col:
                        render_confidence_ring(confidence, is_fake)
                    with met_col:
                        render_metrics(confidence, face_count)

                    render_result_actions("Webcam Capture", "Live", is_fake, confidence, score)

                    add_history("Webcam Capture", "Live", is_fake, confidence, score)

                    if confidence < 0.40:
                        st.warning(
                            "⚠️ **Low Confidence Prediction.** The model is not very certain "
                            "about this result. This can happen with sunglasses, heavy makeup, "
                            "unusual lighting, or non-standard face angles."
                        )

                    if face_count == 0:
                        st.warning(
                            "⚠️ **No faces detected.** The model analyzed the full image. "
                            "For best results, ensure your face is clearly visible and well-lit."
                        )

                    if is_fake:
                        st.toast("🚨 Deepfake detected!", icon="🚨")
                    else:
                        st.toast("✅ Looks authentic!", icon="✅")

                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🔄 Capture Again", use_container_width=True, key="reset_live"):
                        st.session_state.cam_uploader_version += 1
                        st.rerun()
            else:
                st.markdown('<div class="glass-card fade-in" style="text-align:center; padding: 2.5rem 2rem;">', unsafe_allow_html=True)
                if LOTTIE_SHIELD:
                    st_lottie(LOTTIE_SHIELD, height=220, key="live_welcome_shield")
                else:
                    st.markdown("<div style='font-size: 4rem; margin-bottom: 1rem;'>🛡️</div>", unsafe_allow_html=True)
                st.markdown("""
                    <div style="color: #7a7a9e; font-size: 1.1rem; font-weight:600; margin-top:0.5rem;">
                        Awaiting Forensic Core Initialization
                    </div>
                    <div style="color: #4a4a6e; font-size: 0.85rem; margin-top: 0.5rem;">
                        Capture a photo with your webcam for an instant authenticity check
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="powered-by">
        <span class="powered-by-label">Powered by</span>
        <span class="tech-badge">TensorFlow</span>
        <span class="tech-badge">OpenCV</span>
        <span class="tech-badge">MTCNN</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">EfficientNetV2</span>
        <span class="tech-badge">CBAM</span>
    </div>
    <div class="footer">
        <strong>DeepGuard AI</strong> &mdash; AI-Powered Deepfake Detection<br>
        EfficientNetV2-B0 + CBAM Attention &bull; Trained on 140K Real &amp; Fake Faces<br>
        Privacy-First: All analysis runs locally on your machine. No data is uploaded.<br>
        Made with <span class="heart">♥</span> using Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()