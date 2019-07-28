#-*-coding:utf-8-*-

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from threading import Thread

from yolo.preprocess import prep_image, prep_frame, inp_to_image

from opt import opt
import os
import sys
import time
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
                p = Thread(target=self.getitem_yolo, args=())
            else:
                p = mp.Process(target=self.getitem_yolo, args=())
        else:
            raise NotImplementedError
        p.daemon = True
        p.start()
        return self

    def getitem_yolo(self):
        for i in range(self.num_batches):
            img = []
            orig_img = []
            im_name = []
            im_dim_list = []
            for k in range(i*self.batchSize,min(i+1)*self.batchSize,self.datalen):
                inp_dim = int(opt.inp_dim)
                im_name_k = self.imglist[k].rstrip('\n').rstrip('\r')
                im_name_k = os.path.join(self.img_dir, im_name_k)
                img_k, orig_img_k, im_dim_list_k = prep_image(im_name_k, inp_dim)

                img.append(img_k)
                orig_img.append(orig_img_k)
                im_name.append(im_name_k)
                im_dim_list.append(im_dim_list_k)

            with torch.no_grad():
                im = torch.cat(img)
                im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
                im_dim_list = im_dim_list

            while self.Q.full():
                time.sleep(2)

            self.Q.put(((img, orig_img, im_name, im_dim_list)))

    def getitem(self):
        return self.Q.get()

    def length(self):
        return len(self.imglist)

    def len(self):
        return self.Q.qsize()

class DetectionLoader:
    def __int__(self, dataloder, batchSize=1, queueSize=1024):
