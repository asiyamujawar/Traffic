from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

MODEL_NAME = "yolo11n.pt"

# COCO IDs for common traffic-related classes.
TRAFFIC_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
}

class TrafficDetector:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = YOLO(model_name)

    @property
    def class_names(self) -> dict[int, str]:
        names = self.model.names
        return {int(k): str(v) for k, v in names.items()}

    def detect_image(
        self,
        image: Image.Image,
        confidence: float = 0.35,
        selected_classes: list[int] | None = None,
    ) -> tuple[Image.Image, pd.DataFrame, dict[str, int]]:
        rgb = np.array(image.convert("RGB"))
        results = self.model.predict(
            source=rgb,
            conf=confidence,
            classes=selected_classes or None,
            verbose=False,
        )
        result = results[0]
        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        rows: list[dict[str, Any]] = []

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                score = float(box.conf.item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                rows.append({
                    "class_id": cls_id,
                    "class": self.class_names.get(cls_id, str(cls_id)),
                    "confidence": round(score, 4),
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                })

        df = pd.DataFrame(rows, columns=[
            "class_id", "class", "confidence", "x1", "y1", "x2", "y2"
        ])
        counts = dict(Counter(df["class"].tolist())) if not df.empty else {}
        return Image.fromarray(annotated_rgb), df, counts

    def detect_video(
        self,
        input_path: str,
        output_path: str,
        confidence: float = 0.35,
        selected_classes: list[int] | None = None,
        max_frames: int = 1200,
    ) -> tuple[dict[str, int], int, str]:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open this video. Try an MP4 encoded with H.264.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError("Video has invalid dimensions.")

        # MP4 output is broadly supported by the Streamlit browser preview.
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise ValueError("Could not create output video on this system.")

        totals: Counter[str] = Counter()
        frame_count = 0
        try:
            while frame_count < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                result = self.model.predict(
                    source=frame,
                    conf=confidence,
                    classes=selected_classes or None,
                    verbose=False,
                )[0]
                writer.write(result.plot())
                if result.boxes is not None:
                    for cls_id in result.boxes.cls.tolist():
                        totals[self.class_names.get(int(cls_id), str(int(cls_id)))] += 1
                frame_count += 1
        finally:
            cap.release()
            writer.release()

        if frame_count == 0:
            raise ValueError("No frames could be read from the uploaded video.")
        return dict(totals), frame_count, output_path
