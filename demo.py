import torch
from opt import opt
import os
import sys
from tqdm import tqdm
import numpy as np

from dataloader import ImageLoader, DetectionLoader, DetectionProcess, Mscoco, DataWriter
from SPPE.src.main_fast_interence import InferenNet_fast, InferenNet

from fn import getTime

args = opt
args.dataset = 'coco'
# 
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    inputpath = args.inputpath
    inputlist = args.inputlist
    mode = args.mode

    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    if len(inputlist):
        im_names = open(inputlist, 'r').readlines()
    elif len(inputpath) and inputpath != '/':
        for root, dirs, files in os.walk(inputpath):
            im_names = files
    else:
        raise IOError('Error: must contain either --indir/--list')

    data_loader = ImageLoader(im_names=im_names, batchSize=args.detbatch, format='yolo').start()

    print("Lading YOLO model..")
    # 刷新缓冲区
    sys.stdout.flush()
    det_loader = DetectionLoader(dataloder=data_loader, batchSize=args.detbatch).start()
    det_processor = DetectionProcess(detectionLoader=det_loader).start()

    # Load Pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 4 + 1, dataset=pose_dataset)
    else:
        pass
    pose_model.cuda()
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # init data writer
    writer = DataWriter(args.save_video).start()

    data_len = data_loader.length()
    im_names_desc = tqdm(range(data_len))

    batchSize = args.posebatch

    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
            print("人体检测完成")

            # 统计 tensor (张量) 的元素的个数
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                continue

            ckpt_time, det_time = getTime(start_time)
            runtime_profile['dt'].append(det_time)
            # Pose estimation

            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            # heatmap
            hm = []
            for j in range(num_batches):
                inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].cuda()
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            print("姿态估计完成")
            # 将两个张量（tensor）拼接在一起
            hm = torch.cat(hm)
            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pt'].append(post_time)
            hm = hm.cpu()
            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
            # pn 保存图片的时间
            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)

        if args.profile:
            # tqdm
            im_names_desc.set_description(
                'det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
                    dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']),
                    pn=np.mean(runtime_profile['pn']))
            )
    print('=================>模型运行完成')
    if args.save_img or args.save_video:
        print('==============>渲染队列中的剩余图像')
    while(writer.running()):
        pass
    writer.stop()
    final_result = writer.results()
    # 将结果写入json文件









