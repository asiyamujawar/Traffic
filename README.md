# ML-Based Traffic Recognition — Full-Stack Demo

A runnable starter project for recognizing road users from images/video using a pretrained YOLO detector, with a Streamlit UI, live webcam mode, image upload, video upload, confidence filtering, class filtering, annotated output, and downloadable results.

> **Important:** This is a practical prototype, not a certified traffic-safety system. Detection accuracy depends on camera angle, weather, lighting, image quality, and the selected model. Do not use it alone for enforcement, autonomous driving, or safety-critical decisions.

## Features
- Image upload (`jpg`, `jpeg`, `png`, `webp`)
- Video upload (`mp4`, `avi`, `mov`, `mkv`)
- Optional webcam detection
- Pretrained Ultralytics YOLO model (`yolo11n.pt`, downloaded automatically on first run)
- Detection count cards and per-class counts
- Confidence threshold and class filters
- Annotated image/video preview and download
- CSV export of image detections
- Responsive Streamlit interface

## Requirements
- Python 3.10–3.12 recommended
- Internet for first-time model download
- CPU works; NVIDIA GPU is optional

## Windows setup (PowerShell or CMD)
1. Extract the ZIP.
2. Open a terminal in the extracted `ml_traffic_recognition` folder.
3. Create and activate a virtual environment:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, use `python -m venv .venv`.

4. Start the app:

```bat
streamlit run app/main.py
```

5. Open the local URL shown in the terminal, usually `http://localhost:8501`.
6. On first run, Ultralytics downloads the model weights. Keep internet connected for this step.

## Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/main.py
```

## How to use
1. Select **Image**, **Video**, or **Webcam** in the sidebar.
2. Upload a file or enable webcam.
3. Set confidence threshold and choose target classes.
4. Click **Run detection** for image/video.
5. Inspect the annotated result and counts.
6. Download the annotated image/video or CSV results.

## Classes
The default COCO model supports common classes such as person, bicycle, car, motorcycle, bus, and truck. The app focuses on road-related classes by default, but you can choose other detected COCO classes too.

## Project structure
```text
ml_traffic_recognition/
├── app/
│   ├── main.py
│   └── detector.py
├── models/                 # optional place for local model weights
├── data/
│   └── sample_images/
├── requirements.txt
├── .gitignore
└── README.md
```

## Model and licensing notes
- Ultralytics YOLO is provided by the `ultralytics` package and its model weights. Review the current Ultralytics licensing terms before commercial use.
- This project uses a pretrained general-object detector, not a custom-trained traffic-specific model.
- For a stronger deployment, collect representative, permissioned local data; label it; split by location/time (not random near-duplicate frames); fine-tune a model; evaluate precision/recall and per-class errors; and test under rain/night/occlusion conditions.
- The prototype counts detections per frame. It does **not** estimate unique vehicles, vehicle speed, traffic violations, or calibrated traffic flow without additional tracking/calibration logic.
"# Traffic" 
