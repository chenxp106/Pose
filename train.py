from train_pose.src.models.FastPose import Pose
import torch


# def main():
#     input_value1 = torch.rand(1, 3, 224, 224)
#     model = Pose()
#
#     '''
#     torch.Size([1, 256, 56, 56])
#     torch.Size([1, 512, 28, 28])
#     torch.Size([1, 1024, 14, 14])
#     torch.Size([1, 2048, 7, 7])
#     '''
#     print(model(input_value1)[0].shape)
#     print(model(input_value1)[1].shape)
#     print(model(input_value1)[2].shape)
#     print(model(input_value1)[3].shape)
#     # print(model(input_value1))
#
#
# if __name__ == '__main__':
#     main()


# from models.FastPost import createModel
from train_pose.src.models.FastPose import createModel
from opt import opt
from train_pose.src.utils.dataset import coco

import torch
import torch.utils.data
from tqdm import tqdm
from train_pose.src.utils.eval import DataLogger, accuracy
from train_pose.src.utils.img import flip, shuffleLR

from tensorboardX import SummaryWriter
import os

def train(train_loader, m, criterion, optimizer, writer):

    # Logger
    lossLogger = DataLogger()
    accLogger = DataLogger()

    m.train()

    train_loader_desc = tqdm(train_loader)

    for i, (inps, labels, setMask, imgset) in enumerate(train_loader_desc):
        # 自动求导autograd函数功能
        inps = inps.cuda().requires_grad_()
        labels = labels.cuda()
        setMask = setMask.cuda()
        out = m(inps)

        # 计算loss
        loss = criterion(out.mul(setMask), labels)

        # 计算准确率
        acc = accuracy(out.data.mul(setMask), labels.data, train_loader.dataset)
        #
        accLogger.update(acc[0], inps.size(0))
        lossLogger.update(loss.item(), inps.size(0))

        # 加算梯度
        optimizer.zero_grad()
        # 反响传播
        loss.backward()
        optimizer.step()

        opt.trainIters +=1
        # 将数据写道tensorborx
        writer.add_scalar(
            'Train/Loss', lossLogger.avg, opt.trainIters
        )
        writer.add_scalar(
            'Train/Acc', lossLogger.avg, opt.trainIters
        )

        # TQDM
        train_loader_desc.set_description(
            'loss: {loss:.8f} | acc: {acc:.2f}'.format(
                loss=lossLogger.avg,
                acc=accLogger.avg * 100
            )
        )

    train_loader_desc.close()

    return lossLogger.avg, accLogger.avg


def valid(val_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.eval()

    val_loader_desc = tqdm(val_loader)

    for i, (inps, labels, setMask, imgset) in enumerate(val_loader_desc):
        inps = inps.cuda()
        labels = labels.cuda()
        setMask = setMask.cuda()

        with torch.no_grad():
            out = m(inps)

            loss = criterion(out.mul(setMask), labels)

            flip_out = m(flip(inps))
            flip_out = flip(shuffleLR(flip_out, val_loader.dataset))

            out = (flip_out + out) / 2

        acc = accuracy(out.mul(setMask), labels, val_loader.dataset)

        lossLogger.update(loss.item(), inps.size(0))
        accLogger.update(acc[0], inps.size(0))

        opt.valIters += 1

        # Tensorboard
        writer.add_scalar(
            'Valid/Loss', lossLogger.avg, opt.valIters)
        writer.add_scalar(
            'Valid/Acc', accLogger.avg, opt.valIters)

        val_loader_desc.set_description(
            'loss: {loss:.8f} | acc: {acc:.2f}'.format(
                loss=lossLogger.avg,
                acc=accLogger.avg * 100)
        )

    val_loader_desc.close()

    return lossLogger.avg, accLogger.avg


def main():
    # 初始化模型
    m = createModel().cuda()

    # 加载已经训练道模型接着训练
    if opt.loadModel:
        print("加载已训练道模型".format(opt.loadModel))
        m.load_state_dict(torch.load(opt.loadModel))
        pass
    # 新建模型进行训练
    else:
        print('新建一个模型')
        if not os.path.exists("train_pose/exp/{}/{}".format(opt.dataset, opt.expID)):
            try:
                os.mkdir("train_pose/exp/{}/{}".format(opt.dataset, opt.expID))
            except FileNotFoundError:
                os.mkdir("train_pose/exp/{}".format(opt.dataset))
                os.mkdir("train_pose/exp/{}/{}".format(opt.dataset, opt.expID))

    criterion = torch.nn.MSELoss().cuda()
    # 定义优化器
    if opt.optMethod == 'adam':
        optimizer = torch.optim.Adam(
            m.parameters(),
            lr=opt.LR
        )
    elif opt.optMethod == 'rmsprop':
        optimizer = torch.optim.RMSprop(m.parameters(),
                                        lr=opt.LR,
                                        momentum=opt.momentum,
                                        weight_decay=opt.weightDecay)
    else:
        raise Exception

    writer = SummaryWriter(
        '.tensorboard/{}/{}'.format(opt.dataset, opt.expID)
    )

    # 准备训练数据
    if opt.dataset == 'coco':
        train_dataset =coco.Mscoco(train=True)
        val_dataset = coco.Mscoco(train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.trainBatch, shuffle=True, num_workers=opt.nThreads, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.validBatch, shuffle=False, num_workers=opt.nThreads, pin_memory=True
    )

    m = torch.nn.DataParallel(m).cuda()

    # 开始训练
    for i in range(opt.nEpochs):
        opt.epoch = i

        print('############## String Epoch ##########'.format(opt.epoch))
        loss, acc = train(train_loader, m, criterion, optimizer, writer)

        print('Train-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
            idx=opt.epoch,
            loss=loss,
            acc=acc
        ))

        opt.acc = acc
        opt.loss = loss
        m_dev = m.module

        if i % opt.snapshot == 0:
            torch.save(
                m_dev.state_dict(), 'train_pose/exp/{}/{}/model_{}.pkl'.format(opt.dataset, opt.expID, opt.epoch)
            )
            torch.save(
                opt, 'train_pose/exp/{}/{}/option.pkl'.format(opt.dataset, opt.expID, opt.epoch)
            )
            torch.save(
                optimizer, 'train_pose/exp/{}/{}/optimizer.pkl'.format(opt.dataset, opt.expID)
            )

        loss, acc = valid(val_loader, m, criterion, optimizer, writer)

        print(
            'Valid-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
                idx=i,
                loss=loss,
                acc=acc)
        )
    writer.close()



if __name__ == '__main__':
    main()