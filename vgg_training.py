import torch.nn as nn
import torch
import math
import torchvision
from config import device


class VGG(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                      # conv1

                                      nn.Conv2d(64, 128, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 128, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                      # conv2

                                      nn.Conv2d(128, 256, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                                      # conv3

                                      nn.Conv2d(256, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      # conv4

                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                      # conv5
                                      )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 20),
        )

        self.init_conv2d()

        if pretrained:

            std = torchvision.models.vgg16(pretrained=True).features.state_dict()
            model_dict = self.features.state_dict()
            pretrained_dict = {k: v for k, v in std.items() if k in model_dict}  # 여기서 orderdict 가 아니기 때문에
            model_dict.update(pretrained_dict)
            self.features.load_state_dict(model_dict)

    def init_conv2d(self):
        for c in self.features.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    vgg = VGG(pretrained=True).to(device)
    img = torch.FloatTensor(2, 3, 224, 224).to(device)
    print(vgg(img).size())


