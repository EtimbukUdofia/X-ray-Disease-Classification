# app.py
import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import get_model

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

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
    image_path = None
    class_probs = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(image_path)

            # Process and predict
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

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
        image_path=image_path,
        class_probs=class_probs,
        class_labels=CLASSES,
    )


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
