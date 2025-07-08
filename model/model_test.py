import os
import cv2
from ultralytics import YOLO

def test_kidne_model():
    model_path = "C:/Kidney_disease/runs/detect/kidne_model3/weights/best.pt"
    model = YOLO(model_path)
    print("✅ Model loaded successfully.")

    test_image_path = "C:/Users/Danish/Desktop/T1.jpg"
    output_folder = "C:/Users/Danish/Desktop/kidne-test-results"
    os.makedirs(output_folder, exist_ok=True)

    # Load image using OpenCV
    image = cv2.imread(test_image_path)
    image_height, image_width = image.shape[:2]

    # Extract original image name without extension
    image_name = os.path.splitext(os.path.basename(test_image_path))[0]

    results = model(test_image_path)

    tumor_detected = False  # Flag to check if tumor is detected

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        names = model.names  # get class name mapping

        for box, cls_id in zip(boxes, class_ids):
            label = names[cls_id]
            x1, y1, x2, y2 = map(int, box[:4])

            if label.lower() == "tumor":
                tumor_detected = True
                width = x2 - x1
                height = y2 - y1
                tumor_area = width * height
                tumor_percent = (tumor_area / (image_width * image_height)) * 100

                if tumor_percent < 20:
                    stage = "Stage 1 - Small Tumor"
                    advice = "Keep monitoring. Regular check-ups advised."
                elif tumor_percent < 50:
                    stage = "Stage 2 - Medium Tumor"
                    advice = "Consult specialist. Follow-up imaging suggested."
                else:
                    stage = "Stage 3 - Large/Severe Tumor"
                    advice = "Immediate medical attention required!"

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image, stage, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(image, advice, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                # Draw box for kidney or other detected classes
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if not tumor_detected:
        cv2.putText(image, "No tumor detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Generate output file name based on detection
    status_suffix = "_tumor_detected" if tumor_detected else "_no_tumor"
    output_filename = f"{image_name}{status_suffix}.jpg"
    output_image_path = os.path.join(output_folder, output_filename)

    # Save the annotated image
    cv2.imwrite(output_image_path, image)
    print(f"✅ Annotated image saved: {output_image_path}")

if __name__ == "__main__":
    test_kidne_model()
