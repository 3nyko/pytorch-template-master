import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

#NUM_CLASSES = 2 # BENIGN, ATTACK
NUM_CLASSES = 6 # BENIGN, DOS, GAS, RPM, SPEED, STEERING_WHEEL
INPUT_DIM = 8 # DATA_0 až _7

# =====================================================
# =========        CICIoV2024 model           =========
# =====================================================
class IoV_model(BaseModel):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, NUM_CLASSES)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # pokud přijde tensor tvaru [B, 1, 1, N], rozvineme ho
        x = x.view(x.size(0), -1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return x


# =====================================================
# =========           MNIST model             =========
# =====================================================
class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
