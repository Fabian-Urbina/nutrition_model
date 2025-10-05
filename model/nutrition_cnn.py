import torch
import torch.nn as nn
import torch.nn.functional as F

class NutritionCNN(nn.Module):
    def __init__(self):
        super(NutritionCNN, self).__init__()
        # Convoluciones
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        # Después de conv+pool: 64 × 28 × 28 = 50176
        self.fc1 = nn.Linear(50176, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 5)  # 5 macronutrientes

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)  # flatten (batch_size, 50176)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)

        out = self.fc3(x)
        return {"calories": out[:, 0:1],
                "mass":     out[:, 1:2],
                "fat":      out[:, 2:3],
                "carb":     out[:, 3:4],
                "protein":  out[:, 4:5]}
