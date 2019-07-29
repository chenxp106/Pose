#-*-coding:utf-8-*-

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from threading import Thread
from SPPE.src.utils.img import load_image, im_to_torch, cropBox
from SPPE.src.utils.eval import getMultiPeakPrediction

from yolo.preprocess import prep_image, prep_frame, inp_to_image
from yolo.darknet import Darknet
from yolo.util import dynamic_write_results
from multiprocessing import Queue as pQueue
import numpy as np
from SPPE.src.utils.eval import getPrediction
from pPose_nms import pose_nms
from fn import vis_frame

from opt import opt
import os
import sys
import time
import cv2
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
            for k in range(i * self.batchSize, min((i + 1) * self.batchSize, self.datalen)):
                inp_dim = int(opt.inp_dim)
                im_name_k = self.imglist[k].rstrip('\n').rstrip('\r')
                im_name_k = os.path.join(self.img_dir, im_name_k)
                img_k, orig_img_k, im_dim_list_k = prep_image(im_name_k, inp_dim)

                img.append(img_k)
                orig_img.append(orig_img_k)
                im_name.append(im_name_k)
                im_dim_list.append(im_dim_list_k)

            with torch.no_grad():
                # Human Detection
                img = torch.cat(img)
                im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
                im_dim_list_ = im_dim_list

            while self.Q.full():
                time.sleep(2)

            self.Q.put((img, orig_img, im_name, im_dim_list))

    def getitem(self):
        return self.Q.get()

    def length(self):
        return len(self.imglist)

    def len(self):
        return self.Q.qsize()

class DetectionLoader:
    def __init__(self, dataloder, batchSize=1, queueSize=1024):
        self.det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
        self.det_model.load_weights("models/yolo/yolov3-spp.weights")
        self.det_model.net_info['hight'] = opt.inp_dim
        self.det_inp_dim = int(self.det_model.net_info['hight'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        self.det_model.cuda()
        self.det_model.eval()

        self.stopped=False
        self.dataloder = dataloder
        self.batchSize = batchSize
        self.datalen = self.dataloder.length()
        leftover = 0
        if (self.datalen) % batchSize:
            leftover=1
        self.num_batches = self.datalen // batchSize + leftover

        if opt.sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = mp.Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        if opt.sp:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        else:
            p = mp.Process(target=self.update, args=())
            p.daemon = True
            p.start()
        return self


    def update(self):
        for i in range(self.num_batches):
            img, orig_img, im_name, im_dim_list = self.dataloder.getitem()
            if img is None:
                self.Q.put((None, None, None, None, None, None, None))
                return

            # 当网络中的某一个tensor不需要梯度时，可以使用torch.no_grad()来处理
            with torch.no_grad():
                # Human Detction
                img = img.cuda()
                prediction = self.det_model(img, CUDA=True)
                dets = dynamic_write_results(prediction, opt.confidence,
                                             opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
                # mul person
                if isinstance(dets, int) or dets.shape[0] == 0:
                    for k in range(len(orig_img)):
                        if self.Q.full():
                            time.sleep(2)
                        self.Q.put((orig_img[k], im_name[k], None, None, None, None, None))
                    continue
                dets = dets.cpu()
                im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
                scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                # coordinate transfer
                dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2

                dets[:, 1:5] /= scaling_factor
                for j in range(dets.shape[0]):
                    dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                    dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
                boxes = dets[:, 1:5]
                print("boxes", boxes)
                scores = dets[:, 5:6]
                print("scoes", scores)

            for k in range(len(orig_img)):
                boxes_k = boxes[dets[:, 0] == k]
                if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                    if self.Q.full():
                        time.sleep(2)
                    self.Q.put((orig_img[k], im_name[k], None, None, None, None, None))
                    continue
                inps = torch.zeros(boxes_k.size(0), 3, opt.inputResH, opt.inputResW)
                pt1 = torch.zeros(boxes_k.size(0), 2)
                pt2 = torch.zeros(boxes_k.size(0), 2)
                if self.Q.full():
                    time.sleep(2)
                self.Q.put((orig_img[k], im_name[k], boxes_k, scores[dets[:, 0] == k], inps, pt1, pt2))

    def read(self):
        return self.Q.get()

    def len(self):
        return self.Q.qsize()

class DetectionProcess:
    def __init__(self,detectionLoader, queueSize=1024):
        self.detectionLoader = detectionLoader
        self.stopped = False
        self.datalen = self.detectionLoader.datalen

        # initialize the queue used to store data
        if opt.sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = pQueue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        if opt.sp:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        else:
            p = mp.Process(target=self.update, args=())
            p.daemon = True
            p.start()
        return self

    def update(self):
        # keep looping the whole dataset
        for i in range(self.datalen):

            with torch.no_grad():
                (orig_img, im_name, boxes, scores, inps, pt1, pt2) = self.detectionLoader.read()
                if orig_img is None:
                    self.Q.put((None, None, None, None, None, None, None))
                    return
                if boxes is None or boxes.nelement() == 0:
                    while self.Q.full():
                        time.sleep(0.2)
                    self.Q.put((None, orig_img, im_name, boxes, scores, None, None))
                    continue
                inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)

                while self.Q.full():
                    time.sleep(0.2)
                self.Q.put((inps, orig_img, im_name, boxes, scores, pt1, pt2))

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()

def crop_from_dets(img, boxes, inps, pt1, pt2):
    '''
    根据检测结果从原始图像中裁剪人
    :param img:
    :param boxes:
    :param inps:
    :param pt1:
    :param pt2:
    :return:
    '''
    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor(
            (float(box[0]), float(box[1])))
        bottomRight = torch.Tensor(
            (float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]

        scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        try:
            inps[i] = cropBox(tmp_img.clone(), upLeft, bottomRight, opt.inputResH, opt.inputResW)
        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print('===')
        pt1[i] = upLeft
        pt2[i] = bottomRight

    return inps, pt1, pt2

class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/coco/images'  # root image folders
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
        self.nJoints_mpii = 16
        self.nJoints = 33

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

# class DateWriter:
#     '''
#     datawriter
#     '''
#     def __init__(self, save_video=False,
#                  savepath='example/res/1.avi',fourcc = cv2.VideoWriter_fourcc(*'XVID'),fps = 25,frameSize=(640,480),
#                  queueSize=1024):
#         if save_video:
#             self.stream = cv2.VideoWriter(savepath, fourcc, fps, frameSize)
#             assert self.stream.isOpened(), 'Cannot open video for writing'
#
#         self.save_video = save_video
#         self.stopped = False
#         self.final_result = []
#
#         self.Q = Queue(maxsize=queueSize)
#         if opt.save_img:
#             if not os.path.exists(opt.outputpath + '/vis'):
#                 os.mkdir(opt.outputpath + '/vis')
#
#     def start(self):
#         t = Thread(target=self.update, args=())
#         t.daemon = True
#         t.start()
#         return self
#
#     def update(self):
#         # keep looping infinitely
#         while True:
#             # if the thread indicator variable is set, stop the
#             # thread
#             if self.stopped:
#                 if self.save_video:
#                     self.stream.release()
#                 return
#             # otherwise, ensure the queue is not empty
#             if not self.Q.empty():
#                 (boxes, scores, hm_data, pt1, pt2, orig_img, im_name) = self.Q.get()
#                 orig_img = np.array(orig_img, dtype=np.uint8)
#                 if boxes is None:
#                     if opt.save_img or opt.save_video or opt.vis:
#                         img = orig_img
#                         if opt.vis:
#                             cv2.imshow("AlphaPose Demo", img)
#                             cv2.waitKey(30)
#                         if opt.save_img:
#                             cv2.imwrite(os.path.join(opt.outputpath, 'vis', im_name), img)
#
#                         if opt.save_video:
#                             self.stream.write(img)
#                 else:
#                     # location prediction (n, kp, 2) | score prediction (n, kp, 1)
#                     if opt.matching:
#                         pass
#                     else:
#                         preds_hm, preds_img, preds_scores = getPrediction(
#                             hm_data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
#                         result = pose_nms(
#                             boxes, scores, preds_img, preds_scores)
#                     result = {
#                         'imgname': im_name,
#                         'result': result
#                     }
#                     self.final_result.append(result)
#                     if opt.save_img or opt.save_video or opt.vis:
#                         img = vis_frame(orig_img, result)
#                         if opt.vis:
#                             cv2.imshow("AlphaPose Demo", img)
#                             cv2.waitKey(30)
#                         if opt.save_img:
#                             cv2.imwrite(os.path.join(opt.outputpath, 'vis', im_name), img)
#                         if opt.save_video:
#                             self.stream.write(img)
#             else:
#                 time.sleep(0.1)
#
#     def running(self):
#         time.sleep(0.2)
#         return not self.Q.empty()
#
#     def save(self, boxes, scores, hm_data, pt1, pt2, orig_img, im_name):
#         return self.Q.put((boxes, scores, hm_data, pt1, pt2, orig_img, im_name))
#
#     def stop(self):
#         self.stopped = True
#         time.sleep(0.2)
#
#     def result(self):
#         return self.final_result
#
#     def len(self):
#         return self.Q.qsize()


class DataWriter:
    def __init__(self, save_video=False,
                savepath='examples/res/1.avi', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=25, frameSize=(640,480),
                queueSize=1024):
        if save_video:
            # initialize the file video stream along with the boolean
            # used to indicate if the thread should be stopped or not
            self.stream = cv2.VideoWriter(savepath, fourcc, fps, frameSize)
            assert self.stream.isOpened(), 'Cannot open video for writing'
        self.save_video = save_video
        self.stopped = False
        self.final_result = []
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
        if opt.save_img:
            if not os.path.exists(opt.outputpath + '/vis'):
                os.mkdir(opt.outputpath + '/vis')

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                if self.save_video:
                    self.stream.release()
                return
            # otherwise, ensure the queue is not empty
            if not self.Q.empty():
                (boxes, scores, hm_data, pt1, pt2, orig_img, im_name) = self.Q.get()
                orig_img = np.array(orig_img, dtype=np.uint8)
                if boxes is None:
                    if opt.save_img or opt.save_video or opt.vis:
                        img = orig_img
                        if opt.vis:
                            cv2.imshow("AlphaPose Demo", img)
                            cv2.waitKey(30)
                        if opt.save_img:
                            cv2.imwrite(os.path.join(opt.outputpath, 'vis', im_name), img)
                        if opt.save_video:
                            self.stream.write(img)
                else:
                    # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                    if opt.matching:
                        # preds = getMultiPeakPrediction(
                        #     hm_data, pt1.numpy(), pt2.numpy(), opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
                        # result = matching(boxes, scores.numpy(), preds)
                        pass
                    else:
                        preds_hm, preds_img, preds_scores = getPrediction(
                            hm_data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
                        result = pose_nms(
                            boxes, scores, preds_img, preds_scores)
                    result = {
                        'imgname': im_name,
                        'result': result
                    }
                    self.final_result.append(result)
                    if opt.save_img or opt.save_video or opt.vis:
                        print("result", result)
                        img = vis_frame(orig_img, result)
                        if opt.vis:
                            cv2.imshow("AlphaPose Demo", img)
                            cv2.waitKey(30)
                        if opt.save_img:
                            cv2.imwrite(os.path.join(opt.outputpath, 'vis', im_name), img)
                        if opt.save_video:
                            self.stream.write(img)
            else:
                time.sleep(0.1)

    def running(self):
        # indicate that the thread is still running
        time.sleep(0.2)
        return not self.Q.empty()

    def save(self, boxes, scores, hm_data, pt1, pt2, orig_img, im_name):
        # save next frame in the queue
        self.Q.put((boxes, scores, hm_data, pt1, pt2, orig_img, im_name))

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        time.sleep(0.2)

    def results(self):
        # return final result
        return self.final_result

    def len(self):
        # return queue len
        return self.Q.qsize()


