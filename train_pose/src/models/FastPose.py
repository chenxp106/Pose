
from train_pose.src.models.layers.Resnet import resnet50
from train_pose.src.models.layers.Dilated import Dilated
from train_pose.src.models.layers.Deconv import Deconv
import torch.nn as nn
import torch


def createModel():
    return Pose()


class Pose(nn.Module):
    conv_dim = 256
    def __init__(self, pretraine = True):
        super(Pose, self).__init__()
        self.resnet50 = resnet50(pretraine)
        self.dilated = Dilated()
        self.deconv = Deconv()

        self.conv_out = nn.Conv2d(
            self.conv_dim, 17, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        out = self.resnet50(x)
        out = self.dilated(out)
        out = self.deconv(out)

        out = self.conv_out(out)

        return out


if __name__ == '__main__':
    input_value1 = torch.rand(1, 3, 320, 256)
    pose = Pose()
    out = pose(input_value1)
    # print(out[1].shape)
    print(out.shape)