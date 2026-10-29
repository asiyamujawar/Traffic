"""Optional local webcam window. Run from the project root: python app/webcam.py"""
import cv2
from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions or try camera index 1.")
    print("Press q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.predict(frame, conf=0.35, verbose=False)[0]
            cv2.imshow("TrafficVision AI - press q to quit", result.plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
