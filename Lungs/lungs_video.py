import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import torch.nn as nn
from torchvision import models
import os
import csv
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained model
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Predict a single frame
def predict_frame(frame, model):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        prob = torch.softmax(output, dim=1)[0][pred].item()
    classes = ['NORMAL', 'PNEUMONIA']
    return classes[pred.item()], prob

# Annotate a frame with prediction, confidence, and bounding box
def annotate_frame(frame, prediction, confidence, fps):
    text = f"{prediction} ({confidence:.2f}) | FPS: {fps:.1f}"
    font_path = "arial.ttf"
    try:
        font = ImageFont.truetype(font_path, 24)
    except:
        font = ImageFont.load_default()

    # Convert to PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    # Define colors
    text_box_color = (0, 255, 0, 128) if prediction == "NORMAL" else (255, 0, 0, 128)
    lung_box_color = (0, 255, 0) if prediction == "NORMAL" else (255, 0, 0)

    # Draw top-left label box
    draw.rectangle([10, 10, 500, 45], fill=text_box_color)
    draw.text((15, 10), text, fill="white", font=font)

    # Simulated lung bounding box
    w, h = frame.shape[1], frame.shape[0]
    x1, y1 = int(w * 0.25), int(h * 0.3)
    x2, y2 = int(w * 0.75), int(h * 0.8)
    draw.rectangle([x1, y1, x2, y2], outline=lung_box_color, width=4)

    # Draw prediction label above the box
    draw.text((x1 + 5, y1 - 25), prediction, fill=lung_box_color, font=font)

    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Process video with annotation
def process_video(video_path, output_path, model, log_csv_path="predictions_log.csv", skip_frames=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with open(log_csv_path, mode='w', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["Frame", "Prediction", "Confidence"])

        pbar = tqdm(total=frame_count, desc="Processing video")
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames == 0:
                prediction, confidence = predict_frame(frame, model)
                annotated_frame = annotate_frame(frame, prediction, confidence, fps)
                out.write(annotated_frame)
                writer.writerow([frame_idx, prediction, round(confidence, 4)])
            else:
                out.write(frame)

            frame_idx += 1
            pbar.update(1)

        cap.release()
        out.release()
        pbar.close()
        print(f"\nVideo saved with annotations to: {output_path}")
        print(f"Prediction log saved to: {log_csv_path}")

# MAIN
def main():
    model_path = r"C:\Kidney_disease\pneumonia_resnet18.pth"
    video_input_path = r"C:\Users\Danish\Desktop\LUNGS.mp4"
    video_output_path = r"C:\Users\Danish\Desktop\lungs_annotated_output.mp4"
    log_csv_path = r"C:\Users\Danish\Desktop\lungs_prediction_log.csv"

    try:
        model = load_model(model_path)
        process_video(video_input_path, video_output_path, model, log_csv_path=log_csv_path, skip_frames=1)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
