import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from model import CrimeClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_model():
    ckpt = torch.load("checkpoints/best_model.pth", map_location=DEVICE)
    classes = ckpt["classes"]
    model = CrimeClassifier(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, classes

def predict_video(video_path, sample_every=10):
    model, classes = load_model()

    cap = cv2.VideoCapture(video_path)
    preds = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % sample_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            x = tfms(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = model(x)
                p = torch.softmax(out, dim=1)
                pred = torch.argmax(p, dim=1).item()
                preds.append(pred)

        idx += 1

    cap.release()

    if len(preds) == 0:
        return "NoFrames", 0.0

    # majority vote
    pred_class = int(np.bincount(preds).argmax())
    label = classes[pred_class]
    confidence = float(np.mean([1.0 if p == pred_class else 0.0 for p in preds]))

    return label, confidence

if __name__ == "__main__":
    label, conf = predict_video("test.mp4")
    print("Prediction:", label, "Confidence:", conf)
