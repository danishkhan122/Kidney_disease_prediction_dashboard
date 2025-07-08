import tkinter as tk
from tkinter import filedialog
import os
import cv2
from ultralytics import YOLO

def upload_and_detect():
    file_path = filedialog.askopenfilename(title="Select CT Scan Image or Video",
                                           filetypes=[("Media files", "*.jpg *.jpeg *.png *.mp4")])
    if not file_path:
        print("❌ No file selected.")
        return

    model = YOLO("C:/Kidney_disease/runs/detect/kidne_model3/weights/best.pt")
    print("✅ Model loaded.")

    ext = os.path.splitext(file_path)[-1].lower()

    output_folder = "C:/Users/Danish/Desktop/kidne-test-results"
    os.makedirs(output_folder, exist_ok=True)

    if ext in [".jpg", ".jpeg", ".png"]:
        # Process image
        image = cv2.imread(file_path)
        results = model(image)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                w = x2 - x1
                h = y2 - y1
                area = w * h
                percent = (area / (image.shape[0] * image.shape[1])) * 100

                if percent < 20:
                    stage = "Stage 1 - Small Tumor"
                    advice = "Monitor regularly."
                elif percent < 50:
                    stage = "Stage 2 - Medium Tumor"
                    advice = "Consult specialist."
                else:
                    stage = "Stage 3 - Large Tumor"
                    advice = "Urgent medical care!"

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image, stage, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(image, advice, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.imwrite(os.path.join(output_folder, "uploaded_image_annotated.jpg"), image)
        print("✅ Image processed and saved.")

    elif ext == ".mp4":
        # You can reuse your video processing code here
        print("🎥 Processing video...")
        # (copy your video code logic here...)

root = tk.Tk()
root.title("Kidney Report Upload")
tk.Button(root, text="Upload Kidney Report (Image/Video)", command=upload_and_detect).pack(pady=20)
root.mainloop()
