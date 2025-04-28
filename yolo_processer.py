# yolo_processor.py
from ultralytics import YOLO
import cv2
from pathlib import Path

model = YOLO("yolo11n.pt")

def process_video(input_path, output_path="static/output_video.mp4"):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    return str(out_path)

def process_image(input_path, output_path="static/output_image.jpg"):
    image = cv2.imread(input_path)
    results = model(image)
    annotated_image = results[0].plot()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated_image)

    return str(out_path)
