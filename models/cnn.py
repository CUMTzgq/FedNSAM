import torch
import torch.nn as nn
import torch.nn.functional as F


class EMNISTCNN(nn.Module):
    def __init__(self, num_classes: int = 62) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.hidden_layer1 = nn.Linear(1600, 512)
        self.output_layer = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.hidden_layer1(x))
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)


def cnn_emnist(num_classes: int = 62) -> EMNISTCNN:
    return EMNISTCNN(num_classes=num_classes)
