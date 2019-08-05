import torch
import torch.nn as nn
import torch.nn.functional as F


class Dilation(nn.Module):
    def __init__(self, inplanes, outplanes, dilation, padding):
        super(Dilation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=outplanes // 2, kernel_size=1,  bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes // 2)
        self.conv2 = nn.Conv2d(in_channels=outplanes // 2, out_channels=outplanes // 2, kernel_size=3,
                               stride=2, dilation=dilation, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes // 2)
        self.conv3 = nn.Conv2d(in_channels=outplanes // 2, out_channels=outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        residul = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)  # torch.Size([1, 64, 64, 64])
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)  # torch.Size([1, 64, 64, 64])
        out = self.bn3(self.conv3(out))  # torch.Size([1, 128, 64, 64])

        # out +=residul

        out = F.relu(out, inplace=True)

        return out

class Dilated(nn.Module):
    def __init__(self):
        super(Dilated, self).__init__()
        self.layer1 = self._make_layer1(dilation=Dilation)
        self.layer2 = self._make_layer2(dilation=Dilation)
        self.layer3 = self._make_layer3(dilation=Dilation)

    def forward(self, x):
        x1 = self.layer1(x[0])
        x2 = self.layer2(x[1])
        x3 = self.layer3(x[2])
        x4 = x[3]

        return x1+x2+x3+x4
        # return x3

    def _make_layer3(self, dilation):
        layers = []
        layers.append(dilation(1024, 2048, 2, 2))
        return nn.Sequential(*layers)

    def _make_layer2(self, dilation):
        layers = []

        layers.append(dilation(512, 1024, 2, 2))
        layers.append(dilation(1024, 2048, 2, 2))
        return nn.Sequential(*layers)

    def _make_layer1(self, dilation):
        layers = []

        layers.append(dilation(256, 512, 3, 3))
        layers.append(dilation(512, 1024, 3, 3))
        layers.append(dilation(1024, 2048, 3, 3))

        return nn.Sequential(*layers)
