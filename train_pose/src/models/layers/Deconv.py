import torch
import torch.nn as nn


class Deconv(nn.Module):
    def __init__(self):
        super(Deconv, self).__init__()

        self.layer1 = self._make_deconv_layer1()
        self.layer2 = self._make_deconv_layer2()
        self.layer3 = self._make_deconv_layer3()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        return out


    # def _make_deconv_layer(self, num_layer=3, num_filter=[256, 256, 256], num_kernels=3):
    #     layers=[]
    #
    #     for i in range(num_layer):
    #         kernel = 3
    #         padding = 1
    #         putput_padding = 1
    #         plans = num_filter[i]
    #         layers.append(
    #             nn.ConvTranspose2d(
    #                 in_channels=self.
    #             )
    #         )
    def _make_deconv_layer1(self):
        layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=2048,
                out_channels=1024,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        return layer1

    def _make_deconv_layer2(self):
        layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        return layer2

    def _make_deconv_layer3(self):
        layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        return layer3

if __name__ == '__main__':
    d = Deconv()
    input_value1 = torch.rand(1, 2048, 7, 7)
    out = d(input_value1)
    print(out.shape)