#-*-coding:utf-8-*-

import torch
from opt import opt
import numpy as np
from SPPE.src.utils.img import transformBoxInvert_batch

def getPrediction(hms, pt1, pt2, inpH, inpW, resH, resW):
    '''
    Get keypoint location from heatmaps
    '''

    assert hms.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(hms.view(hms.size(0), hms.size(1), -1), 2)

    maxval = maxval.view(hms.size(0), hms.size(1), 1)
    idx = idx.view(hms.size(0), hms.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % hms.size(3)
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / hms.size(3))

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask

    # Very simple post-processing step to improve performance at tight PCK thresholds
    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm = hms[i][j]
            pX, pY = int(round(float(preds[i][j][0]))), int(round(float(preds[i][j][1])))
            if 0 < pX < opt.outputResW - 1 and 0 < pY < opt.outputResH - 1:
                diff = torch.Tensor(
                    (hm[pY][pX + 1] - hm[pY][pX - 1], hm[pY + 1][pX] - hm[pY - 1][pX]))
                preds[i][j] += diff.sign() * 0.25
    preds += 0.2

    preds_tf = torch.zeros(preds.size())

    preds_tf = transformBoxInvert_batch(preds, pt1, pt2, inpH, inpW, resH, resW)

    return preds, preds_tf, maxval

#################
def getMultiPeakPrediction(hms, pt1, pt2, inpH, inpW, resH, resW):

    assert hms.dim() == 4, 'Score maps should be 4-dim'

    preds_img = {}
    hms = hms.numpy()
    for n in range(hms.shape[0]):        # Number of samples
        preds_img[n] = {}           # Result of sample: n
        for k in range(hms.shape[1]):    # Number of keypoints
            preds_img[n][k] = []    # Result of keypoint: k
            hm = hms[n][k]

            candidate_points = findPeak(hm)

            res_pt = processPeaks(candidate_points, hm,
                                  pt1[n], pt2[n], inpH, inpW, resH, resW)

            preds_img[n][k] = res_pt

    return preds_img