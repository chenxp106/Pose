import torch
from opt import opt

from pycocotools.coco import  COCO
from pycocotools.cocoeval import COCOeval


class DataLogger(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        self.avg = self.sum / self.cnt


def accuracy(output, label, dataset, out_offset=None):
    if type(output) == list:
        return accuracy(output[opt.nStack - 1], label[opt.nStack - 1], dataset, out_offset)
    else:
        return heatmapAccuracy(output.cpu().data, label.cpu().data, dataset.accIdxs)


def heatmapAccuracy(output, label, idxs):
    preds = getPreds(output)
    gt = getPreds(label)

    norm = torch.ones(preds.size(0)) * opt.outputResH / 10
    dists = calc_dists(preds, gt, norm)

    acc = torch.zeros(len(idxs) + 1)
    avg_acc = 0
    cnt = 0
    for i in range(len(idxs)):
        acc[i + 1] = dist_acc(dists[idxs[i] - 1])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1
    if cnt != 0:
        acc[0] = avg_acc / cnt
    return acc


def getPreds(hm):
    '''
    从heatmap中得到预测的结果
    :param hm:
    :return:
    '''
    assert hm.dim() == 4, 'Scope map是四维的'
    maxval, idx = torch.max(hm.view(hm.size(0), hm.size(1), -1), 2)

    maxval = maxval.view(hm.size(0), hm.size(1), 1)
    idx = idx.view(hm.size(0), hm.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % hm.size(3)
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / hm.size(3))

    return preds


def calc_dists(preds, target, normalize):
    '''
    计算真实与预测的距离
    :param preds:
    :param target:
    :param normalize:
    :return:
    '''

    preds = preds.float().clone()
    target = target.float().clone()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n, c, 0] > 0 and target[n, c, 1] > 0:
                # 计算欧式距离
                dists[c, n] = torch.dist(
                    preds[n, c, :], target[n, c, :]) / normalize[n]
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    '''
    计算得分介于0-1之间
    :param dists:
    :param thr:
    :return:
    '''
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).float().sum() * 1.0 / dists.ne(-1).float().sum()
    else:
        return - 1