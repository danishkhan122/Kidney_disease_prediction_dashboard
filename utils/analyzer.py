import numpy as np
import cv2
from ultralytics import YOLO

def analyze_image(img_pil):
    model = YOLO("assets/best.pt")
    img_np = np.array(img_pil.convert("RGB"))
    annotated = img_np.copy()
    results = model(img_np)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    names = model.names

    kidneys, tumors = [], []
    stage_counts = [0, 0, 0]
    h, w = img_np.shape[:2]

    for box, cls_id in zip(boxes, class_ids):
        label = names[cls_id].lower()
        if label == "kidney":
            kidneys.append(box)
        elif label == "tumor":
            tumors.append(box)

    tumor_matched = [False] * len(kidneys)
    overlay = annotated.copy()
    alpha = 0.3

    for tumor_box in tumors:
        x1, y1, x2, y2 = map(int, tumor_box[:4])
        tumor_area = (x2 - x1) * (y2 - y1)
        tumor_percent = (tumor_area / (w * h)) * 100

        for i, kidney_box in enumerate(kidneys):
            if is_inside(tumor_box, kidney_box) or boxes_intersect(tumor_box, kidney_box):
                tumor_matched[i] = True
                break

        stage = (
            "Stage 1" if tumor_percent < 20 else
            "Stage 2" if tumor_percent < 50 else
            "Stage 3"
        )
        if stage == "Stage 1": stage_counts[0] += 1
        if stage == "Stage 2": stage_counts[1] += 1
        if stage == "Stage 3": stage_counts[2] += 1

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)

    for i, kidney_box in enumerate(kidneys):
        x1, y1, x2, y2 = map(int, kidney_box[:4])
        color = (0, 0, 255) if tumor_matched[i] else (0, 255, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
    return annotated, len(kidneys), len(tumors), stage_counts

def is_inside(boxA, boxB):
    Ax1, Ay1, Ax2, Ay2 = boxA
    Bx1, By1, Bx2, By2 = boxB
    return Ax1 >= Bx1 and Ay1 >= By1 and Ax2 <= Bx2 and Ay2 <= By2

def boxes_intersect(boxA, boxB):
    Ax1, Ay1, Ax2, Ay2 = boxA
    Bx1, By1, Bx2, By2 = boxB
    return not (Ax2 < Bx1 or Ax1 > Bx2 or Ay2 < By1 or Ay1 > By2)
