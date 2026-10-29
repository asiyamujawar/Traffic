from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from detector import TrafficDetector, TRAFFIC_CLASSES

st.set_page_config(
    page_title="TrafficVision AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background: #0b1020; color: #eef2ff; }
  [data-testid="stSidebar"] { background: #10182b; }
  .hero {
    padding: 1.5rem 1.7rem; border-radius: 18px;
    background: linear-gradient(120deg, #18294a, #183d4b);
    border: 1px solid #29435f; margin-bottom: 1.25rem;
  }
  .hero h1 { margin: 0; font-size: 2.15rem; color: #f8fafc; }
  .hero p { color: #c6d6e8; margin: .45rem 0 0 0; }
  .small-note { color: #9fb0c7; font-size: .9rem; }
  div[data-testid="stMetric"] {
    background: #111b30; border: 1px solid #263752;
    padding: 14px; border-radius: 14px;
  }
  .stButton > button { border-radius: 10px; font-weight: 650; }
</style>
<div class="hero">
  <h1>🚦 TrafficVision AI</h1>
  <p>Machine-learning powered road-user detection from images and video.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading the YOLO model (first run may download weights)...")
def load_detector() -> TrafficDetector:
    return TrafficDetector()

with st.sidebar:
    st.header("Detection settings")
    mode = st.radio("Input mode", ["Image", "Video", "Webcam"], index=0)
    confidence = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
    traffic_only = st.toggle("Focus on traffic-related classes", value=True)
    st.caption("Higher confidence reduces weaker detections but may miss objects.")
    st.divider()
    st.caption("Model: YOLO11 nano · COCO pretrained")
    st.caption("Runs locally after dependencies and model weights are installed.")

st.markdown(
    "Upload a road scene, adjust the confidence threshold, and run detection. "
    "Results are indicative and should be checked by a human."
)

try:
    detector = load_detector()
except Exception as exc:
    st.error("The detector could not be loaded.")
    st.code(str(exc))
    st.info("Check your internet connection for the initial model download, then run `pip install -r requirements.txt` again.")
    st.stop()

names = detector.class_names
if traffic_only:
    default_ids = [i for i in TRAFFIC_CLASSES if i in names]
else:
    default_ids = list(names.keys())

with st.sidebar:
    selected_names = st.multiselect(
        "Classes to detect",
        options=[names[i] for i in sorted(names)],
        default=[names[i] for i in default_ids],
    )
selected_ids = [i for i, name in names.items() if name in selected_names]

if mode == "Image":
    uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Original image")
            st.image(image, use_container_width=True)
        if st.button("Run detection", type="primary", use_container_width=True):
            with st.spinner("Detecting road users..."):
                annotated, df, counts = detector.detect_image(image, confidence, selected_ids)
            st.session_state["traffic_image_result"] = (annotated, df, counts)
        if "traffic_image_result" in st.session_state:
            annotated, df, counts = st.session_state["traffic_image_result"]
            with right:
                st.subheader("Detection result")
                st.image(annotated, use_container_width=True)
                png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                annotated.save(png.name)
                st.download_button(
                    "Download annotated image",
                    data=Path(png.name).read_bytes(),
                    file_name="traffic_detection.png",
                    mime="image/png",
                    use_container_width=True,
                )
            st.divider()
            st.subheader("Detection summary")
            a, b, c = st.columns(3)
            a.metric("Objects detected", len(df))
            b.metric("Classes found", len(counts))
            c.metric("Confidence threshold", f"{confidence:.0%}")
            if counts:
                st.bar_chart(pd.DataFrame(
                    [{"Class": k, "Detections": v} for k, v in counts.items()]
                ).set_index("Class"))
            else:
                st.info("No objects matched the current classes and confidence threshold.")
            st.subheader("Bounding-box details")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download detections as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="traffic_detections.csv",
                mime="text/csv",
            )
    else:
        st.info("Choose an image file above to begin.")

elif mode == "Video":
    uploaded = st.file_uploader("Upload a road video", type=["mp4", "avi", "mov", "mkv"])
    max_frames = st.slider("Maximum frames to process", 100, 1200, 600, 100)
    st.caption("For speed, the app processes up to the selected number of frames.")
    if uploaded and st.button("Run video detection", type="primary"):
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as source:
            source.write(uploaded.getbuffer())
            source_path = source.name
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        try:
            with st.spinner("Processing video frames..."):
                counts, frames, result_path = detector.detect_video(
                    source_path, output_path, confidence, selected_ids, max_frames
                )
            st.success(f"Processed {frames} frames.")
            st.video(Path(result_path).read_bytes())
            a, b = st.columns(2)
            a.metric("Frames processed", frames)
            b.metric("Detection instances", sum(counts.values()))
            if counts:
                st.bar_chart(pd.DataFrame(
                    [{"Class": k, "Detections": v} for k, v in counts.items()]
                ).set_index("Class"))
            st.download_button(
                "Download annotated video",
                data=Path(result_path).read_bytes(),
                file_name="traffic_detection.mp4",
                mime="video/mp4",
            )
        except Exception as exc:
            st.error(f"Video processing failed: {exc}")
        finally:
            Path(source_path).unlink(missing_ok=True)

else:
    st.warning("Webcam mode uses your browser's camera permission and may not work in every deployment.")
    st.markdown("For a reliable local webcam stream, use the optional OpenCV script described below.")
    if st.button("Start local webcam detector", type="primary"):
        st.info("Webcam access through Streamlit is browser-dependent. For best results, use `python app/webcam.py` after installing dependencies.")

st.divider()
st.markdown(
    '<p class="small-note">Prototype only · No identity recognition · No automatic enforcement · '
    'Detection totals are instances, not unique vehicles.</p>',
    unsafe_allow_html=True,
)
