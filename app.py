# app.py
import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import get_model
from io import BytesIO
import base64

app = Flask(__name__)

# Your 6 class labels
CLASSES = [
    "Bacterial Pneumonia",
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia",
]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=len(CLASSES))
model.load_state_dict(torch.load("model/xray_model.pth", map_location=device))
model.to(device)
model.eval()

# Image transform (RGB input)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_data = None
    class_probs = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            image_bytes = file.read()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            image_data = base64.b64encode(image_bytes).decode("utf-8")

            with torch.no_grad():
                output = model(image)
                probabilities = F.softmax(output, dim=1)
                confidence_score = torch.max(probabilities).item() * 100
                _, predicted = torch.max(output, 1)
                prediction = CLASSES[predicted.item()]
                confidence = round(confidence_score, 2)
                class_probs = [
                    round(p * 100, 2) for p in probabilities.squeeze().tolist()
                ]

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_data=image_data,
        class_probs=class_probs,
        class_labels=CLASSES,
    )


@app.route("/ping")
def ping():
    return "pong", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
