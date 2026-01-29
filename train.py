import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from utils import seed_everything, list_classes
from video_dataset import UCFFrameDataset
from model import CrimeClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    seed_everything(42)

    # ✅ Speed boost for NVIDIA GPU
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    train_dir = "dataset/train"
    val_dir = "dataset/val"

    classes = list_classes(train_dir)
    num_classes = len(classes)

    # ✅ Confirm GPU usage
    print("✅ Device:", DEVICE)
    if DEVICE == "cuda":
        print("✅ GPU:", torch.cuda.get_device_name(0))
        print("✅ Torch CUDA:", torch.version.cuda)

    print("✅ Classes:", classes)
    print("✅ Num classes:", num_classes)

    train_ds = UCFFrameDataset(train_dir, classes)
    val_ds = UCFFrameDataset(val_dir, classes)

    BATCH_SIZE = 32  # if out of memory -> set 16

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = CrimeClassifier(num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ✅ Mixed Precision for faster training on RTX 3050
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    os.makedirs("checkpoints", exist_ok=True)

    best_acc = 0.0
    EPOCHS = 10

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # ✅ Forward with AMP
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            # ✅ Backward with AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # ✅ Validation
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                    outputs = model(imgs)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(labels.numpy())

        acc = accuracy_score(y_true, y_pred)
        avg_loss = total_loss / len(train_loader)

        print(f"\nEpoch {epoch} | Loss={avg_loss:.4f} | ValAcc={acc:.4f}\n")

        # ✅ Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(
                {"model": model.state_dict(), "classes": classes},
                "checkpoints/best_model.pth"
            )
            print("✅ Saved best_model.pth")

    print("✅ Training Done! Best ValAcc =", best_acc)
    print("\nFinal Validation Report:\n")
    print(classification_report(y_true, y_pred, target_names=classes))


if __name__ == "__main__":
    train()
