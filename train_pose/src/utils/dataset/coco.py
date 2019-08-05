#-*-coding:utf-8-*-
import h5py
import torch.utils.data as data
from opt import opt
from functools import reduce
from ..pose import generateSampleBox

import os


class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '/media/ubuntu/文档/code/torch/AlphaPose-pytorch/train_sppe/data/coco/images/train2017'  # root image folders
        self.is_train = train  # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 17
        self.nJoints = 17

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))

        # 分割训练集和测试集
        with h5py.File('train_pose/data/coco/annot_coco.h5', 'r') as annot:
            self.imgname_coco_train = annot['imgname'][:20]
            self.bndbox_coco_train = annot['bndbox'][:20]
            self.part_coco_train = annot['part'][:20]
            # val
            self.imgname_coco_val = annot['imgname'][0:1]
            self.bndbox_coco_val = annot['bndbox'][0:1]
            self.part_coco_val = annot['part'][0:1]

        self.size_train = self.imgname_coco_train.shape[0]
        self.size_val = self.imgname_coco_val.shape[0]

    def __getitem__(self, index):
        sf = self.scale_factor

        if self.is_train:
            part = self.part_coco_train[index]
            bndbox = self.bndbox_coco_train[index]
            imgname = self.imgname_coco_train[index]

        else:
            part = self.part_coco_val[index]
            bndbox = self.bndbox_coco_val[index]
            imgname = self.imgname_coco_val[index]

        imgname = reduce(lambda x, y: x + y,
                         map(lambda x: chr(int(x)), imgname))
        img_path = os.path.join(self.img_folder, imgname)

        metaData = generateSampleBox(img_path, bndbox, part, self.nJoints,
                                     'coco', sf, self, train=self.is_train)

        inp, out, setMask = metaData

        return inp, out, setMask, 'coco'

    def __len__(self):
        if self.is_train:
            return self.size_train

        else:
            return self.size_val