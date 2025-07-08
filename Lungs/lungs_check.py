import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import torch.nn as nn
from torchvision import models
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load the trained model
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Assuming 2 classes (e.g., Normal, Diseased)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# Image preprocessing (same as training script)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# Predict function
def predict_image(image_path, model):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)
            confidence = probabilities[0][predicted].item()

        # Class names (adjust if different for kidney disease)
        class_names = ['NORMAL', 'PNEUMONIA']  # Update if your classes differ
        prediction = class_names[predicted.item()]

        return prediction, confidence, image
    except FileNotFoundError:
        return f"Error: Image file '{image_path}' not found", None, None
    except Exception as e:
        return f"Error: {str(e)}", None, None


# Function to annotate and save image
def annotate_and_save_image(image, prediction, confidence, output_path):
    try:
        draw = ImageDraw.Draw(image)

        # Try to use a default font, fall back to basic if unavailable
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        # Define text and position
        text = f"Prediction: {prediction}, Confidence: {confidence:.4f}" if confidence is not None else prediction
        text_position = (10, 10)  # Top-left corner

        # Add a semi-transparent background rectangle for text readability
        text_bbox = draw.textbbox(text_position, text, font=font)
        draw.rectangle(text_bbox, fill=(0, 0, 0, 128))  # Black with 50% opacity

        # Draw text
        draw.text(text_position, text, fill="white", font=font)

        # Save annotated image
        image.save(output_path)
        print(f"Annotated image saved to: {output_path}")

    except Exception as e:
        print(f"Error saving annotated image: {str(e)}")


def main():
    # Paths from user input
    model_path = r"C:\Kidney_disease\pneumonia_resnet18.pth"  # Adjust to your actual model file name
    image_path = r"C:\Users\Danish\Desktop\lungs1.jpg"  # Provided image path
    output_image_path = r"C:\Users\Danish\Desktop\lungs_annotated.jpeg"  # Output path for annotated image

    try:
        # Load model
        model = load_model(model_path)

        # Make prediction
        prediction, confidence, original_image = predict_image(image_path, model)

        if confidence is not None:
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.4f}")

            # Annotate and save image
            annotate_and_save_image(original_image, prediction, confidence, output_image_path)
        else:
            print(prediction)  # Error message

    except FileNotFoundError:
        print(f"Error: Could not find model file '{model_path}'")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == '__main__':
    main()