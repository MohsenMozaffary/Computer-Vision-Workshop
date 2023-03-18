import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # remove last layer
        self.resnet = nn.Sequential(*modules)
        num_features = resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        x = self.resnet(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x.float()))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x