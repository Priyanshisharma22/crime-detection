import cv2
import torch
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

def main():
    model, classes = load_model()

    cap = cv2.VideoCapture(0)
    frame_id = 0
    label = "..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # predict every 10 frames
        if frame_id % 10 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            x = tfms(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = model(x)
                prob = torch.softmax(out, dim=1)[0]
                pred = torch.argmax(prob).item()
                label = f"{classes[pred]} ({prob[pred].item():.2f})"

        cv2.putText(frame, label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("CCTV Crime Detection (Frame Based)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
