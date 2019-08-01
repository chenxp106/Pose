
import torch
import torch.nn as nn
from SPPE.src.models.FastPose import createModel
from SPPE.src.utils.img import flip, shuffleLR


class InferenNet_fast(nn.Module):
    def __init__(self, kernel_size, dataset):
        super(InferenNet_fast, self).__init__()

        model = createModel().cuda()
        print('加载模型 {}'.format('./models/sppe/duc_se.pth'))

        # model.load_state_dict(torch.load('./models/sppe/duc_se.pth'))
        model.load_state_dict(torch.load('/media/ubuntu/文档/code/Pose/train_sppe/exp/coco/exp5/model_4.pkl'))
        model.eval()
        self.pyranet = model
        self.dataset = dataset

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, 17)

        return out