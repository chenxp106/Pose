import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)  # torch.Size([1, 64, 112, 112])
        out = self.bn1(out)  # torch.Size([1, 64, 112, 112])
        out = self.relu(out)  # torch.Size([1, 64, 112, 112])
        out = self.maxpool(out)  # torch.Size([1, 64, 56, 56])

        x1 = self.layer1(out)  # torch.Size([1, 256, 56, 56]) stride=1,尺度大小没有改变

        x2 = self.layer2(x1)  # torch.Size([1, 512, 28, 28]) stride = 2,尺度大小变成了一半。同时通道数翻倍，变成512

        x3 = self.layer3(x2)  # torch.Size([1, 1024, 14, 14]) stride = 2, 尺度变成原来的一半，通道数翻倍，变成1024

        x4 = self.layer4(x3)  # torch.Size([1, 2048, 7, 7])  stride = 2, 尺度变成原来的一半，通道数翻倍，变成2048

        return [x1, x2, x3, x4]


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        from collections import OrderedDict
        state_dict = model.state_dict()
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        for k, v in pretrained_state_dict.items():
            if k not in state_dict:
                continue
            state_dict[k] = v
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    #
    input_value1 = torch.rand(1, 3, 224, 224)
    # input_value2 = torch.rand(1, 64, 64, 64)
    # input_value3 = torch.rand(1, 512, 64, 64)
    # input_value4 = torch.rand(1, 512, 64, 64)
    #
    # b = Bottleneck(512, 128, 1)
    # out = b(input_value3)
    # print(out.shape)
    # input_value1 = torch.rand(1, 3, 224, 224)
    # r = ResNet(block=Bottleneck, layers=[3, 4, 6, 3])
    # out = r(input_value1)
    # print(out.shape)
    model = resnet50(pretrained=True)
    out = model(input_value1)
    print(out.shape)