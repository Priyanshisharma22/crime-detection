import torch.nn as nn
from torchvision.models import resnet18

class CrimeClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(weights="DEFAULT")
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
