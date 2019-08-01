
import torch.nn as nn
from models.layers.SE_module import SELayer
import torch.nn.functional as F


# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, plans, stride=1, downsample=None, reduction=True):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, plans, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(plans, momentum=0.1)
#         self.conv2 = nn.Conv2d(plans, plans, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(plans, momentum=0.1)
#         self.conv3 = nn.Conv2d(plans, plans * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(plans * 4, momentum=0.1)
#         if reduction:
#             self.se = SELayer(plans * 4)
#
#         self.reduc = reduction
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = F.relu(self.bn1(self.conv1(x)), inplace=True)
#         out = F.relu(self.bn2(self.conv2(out)), inplace=True)
#
#         out = F.relu(self.bn3(self.conv3(out)))
#
#         if self.reduc:
#             out = self.se(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = F.relu(out)
#
#         return out
#
#
# class SEResnet(nn.Module):
#     '''SEResnet'''
#
#     def __init__(self, architecture):
#         super(SEResnet, self).__init__()
#         assert architecture in ["resnet50", "resnet101"]
#         self.inplanes = 64
#         self.layers = [3, 4, {"resnet50": 6, "resnet101": 5}[architecture], 3]
#         self.block = Bottleneck
#
#         self.conv1 = nn.Conv2d(
#             3, 64, kernel_size=7,
#             stride=2, padding=3, bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.layer1 = self.make_layer(self.block, 64, self.layers[0])
#         self.layer2 = self.make_layer(
#             self.block, 128, self.layers[1], stride=2
#         )
#         self.layer3 = self.make_layer(
#             self.block, 256, self.layers[2], stride=2
#         )
#         self.layer4 = self.make_layer(
#             self.block, 512, self.layers[3], stride=2
#         )
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         return x
#
#     def make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
#             )
#
#         layers = []
#         if downsample is not None:
#             layers.append(block(self.inplanes, planes, stride, downsample, reduction=True))
#         else:
#             layers.append(block(self.inplanes, planes, stride, downsample))
#
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def stage(self):
#         return [self.layer1, self.layer2, self.layer3, self.layer4]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,inplanes, planes, stride=1, downsample=None, reduction=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # Bottleneck的conv3会将输入的通道数扩展成原来的4倍，导致输入一定和输出尺寸不同。
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        if reduction:
            self.se = SELayer(planes * 4)

        self.reduc = reduction
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.reduc:
            out = self.se(out)

        # 这个用1x1 的卷积将x维度映射道高维度，然后才能相加。
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


class SEResnet(nn.Module):
    '''
    SERnet
    '''
    def __init__(self, architecture):
        super(SEResnet, self).__init__()
        assert architecture in ["resnet50", "resnet101"]
        self.inplanes = 64
        self.layers = [3, 4, {"resnet50": 6, "resnet101": 23}[architecture], 3]
        self.block = Bottleneck

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        # 2.eps：分母中添加的一个值，目的是为了计算的稳定性，默认为：1e-5
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(self.block, 64, self.layers[0])
        self.layer2 = self.make_layer(
            self.block, 128, self.layers[1], stride=2
        )
        self.layer3 = self.make_layer(
            self.block, 256, self.layers[2], stride=2
        )
        self.layer4 = self.make_layer(
            self.block, 512, self.layers[3], stride=2
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 64 * h/4 * w/4

        x = self.layer1(x)  # 256 * h/4 * w/4
        x = self.layer2(x)  # 512 * h/8 * w/8
        x = self.layer3(x)  # 1024 * h/16 * w/16
        x = self.layer4(x)  # 2048 * h/32 * w/32

        return x

    def stage(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        # 扩维度
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        # 如果需要扩维度，则将reducetion设置为True
        if downsample is not None:
            layers.append(block(self.inplanes, planes, stride, downsample, reduction=True))
        # 第一个惨差模块不需要扩维度，可以直接相加
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        # 循环多个惨差模块
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
