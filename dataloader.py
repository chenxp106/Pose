#-*-coding:utf-8-*-

import torch.utils.data as data
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from threading import Thread

from opt import opt
import os
import sys
if sys.version_info >= (3, 0):
    from queue import Queue,LifoQueue

class Image_loader(data.Dataset):
    def __init__(self,im_names,format='yolo'):
        super(Image_loader,self).__init__()
        self.img_dir = opt.inputpath
        self.imglist = im_names
        # 对输入图像img循环所有的transform操作
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        #
        self.format = format

    def getitem_yolo(self,index):
        inp_dim = int(opt.inp_dim)
        im_name = self.imglist[index].rstrip('\n').rstrip('\r')
        im_name = os.path.join(self.img_dir,im_name)


    def __getitem__(self, index):
        if self.format == 'yolo':
            return self.getitem_yolo(index)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.imglist)


class ImageLoader:
    def __init__(self, im_names, batchSize = 1, format='yolo', queusSize = 50):
        self.img_dir  = opt.inputpath
        self.imglist = im_names
        # 对输入图像img循环所有的transform操作
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.format = format

        self.batchSize = batchSize
        self.datalen = len(self.imglist)
        leftover = 0
        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover

        if opt.sp:
            self.Q = Queue(maxsize=queusSize)
        else:
            self.Q = mp.Queue(maxsize=queusSize)

    def start(self):
        if self.format == 'yolo':
            if opt.sp:
                p = Thread(target=self.getitem_yolo,args=())
            else:
                p = mp.Process(target=self.getitem_yolo,args=())
        else:
            raise NotImplementedError
        p.daemon = True
        p.start()
        return self

    def getitem_yolo(self):
        for i in range(self.num_batches):
            img = []
            origin_img = []
            im_name = []
            im_dim_list = []
            for k in range(i*self.batchSize,min(i+1)*self.batchSize,self.datalen):
                pass

ina = ImageLoader()
ina.start()