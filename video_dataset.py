import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class UCFFrameDataset(Dataset):
    def __init__(self, root_dir, classes, img_size=224):
        self.root_dir = root_dir
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = []

        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for file in os.listdir(cls_dir):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, file), self.class_to_idx[cls]))

        self.tfms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.tfms(img)
        return img, torch.tensor(label, dtype=torch.long)
