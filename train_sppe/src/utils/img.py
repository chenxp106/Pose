import numpy as np
import torch
from PIL import Image
import cv2

import matplotlib.pyplot as plt


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def torch_to_im(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # C*H*W
    return img


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def load_image(img_path):
    return im_to_torch(Image.open(img_path).convert('RGB'))


def drawGaussian(img, pt, sigma):
    '''
    画高斯分布道label
    :param img: img
    :param pt: 关键点道位置
    :param sigma: sigma
    :return:
    '''
    img = to_numpy(img)
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
    br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    sigma = size / 4.0
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img)


def transformBox(pt, ul, br, inpH, inpW, resH, resW):
    '''
    将关键点的位置移到相应的位置上
    :param pt:
    :param ul:
    :param br:
    :param inpH:
    :param inpW:
    :param resH:
    :param resW:
    :return:
    '''
    center = torch.zeros(2)
    center[0] = (br[0] - 1 - ul[0]) / 2
    center[1] = (br[1] - 1 - ul[1]) / 2

    lenH = max(br[1] - ul[1], (br[0] - ul[0]) * inpH / inpW)
    lenW = lenH * inpW / inpH

    _pt = torch.zeros(2)
    _pt[0] = pt[0] - ul[0]
    _pt[1] = pt[1] - ul[1]
    # Move to center
    _pt[0] = _pt[0] + max(0, (lenW - 1) / 2 - center[0])
    _pt[1] = _pt[1] + max(0, (lenH - 1) / 2 - center[1])
    pt = (_pt * resH) / lenH
    pt[0] = round(float(pt[0]))
    pt[1] = round(float(pt[1]))
    return pt.int()


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def cropBox(img, ul, br, resH, resW):
    '''
    根据坐标点裁剪对应长宽道图片
    :param img:
    :param ul:
    :param br:
    :param resH:
    :param resW:
    :return:
    '''
    ul = ul.int()
    br = (br - 1).int()
    # br = br.int()
    lenH = max((br[1] - ul[1]).item(), (br[0] - ul[0]).item() * resH / resW)
    lenW = lenH * resW / resH
    if img.dim() == 2:
        img = img[np.newaxis, :]

    box_shape = [br[1] - ul[1], br[0] - ul[0]]
    pad_size = [(lenH - box_shape[0]) // 2, (lenW - box_shape[1]) // 2]
    # Padding Zeros
    img[:, :ul[1], :], img[:, :, :ul[0]] = 0, 0
    img[:, br[1] + 1:, :], img[:, :, br[0] + 1:] = 0, 0

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = np.array([ul[0] - pad_size[1], ul[1] - pad_size[0]], np.float32)
    src[1, :] = np.array([br[0] + pad_size[1], br[1] + pad_size[0]], np.float32)
    dst[0, :] = 0
    dst[1, :] = np.array([resW - 1, resH - 1], np.float32)

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    dst_img = cv2.warpAffine(torch_to_im(img), trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)

    return im_to_torch(torch.Tensor(dst_img))


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def cv_rotate(img, rot, resW, resH):

    center = np.array((resW - 1, resH - 1)) / 2
    rot_rad = np.pi * rot / 180

    src_dir = get_dir([0, (resH - 1) * -0.5], rot_rad)
    dst_dir = np.array([0, (resH - 1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [(resW - 1) * 0.5, (resH - 1) * 0.5]
    dst[1, :] = np.array([(resW - 1) * 0.5, (resH - 1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    dst_img = cv2.warpAffine(torch_to_im(img), trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)

    return im_to_torch(torch.Tensor(dst_img))


def shuffleLR(x, dataset):
    flipRef = dataset.flipRef
    assert (x.dim() == 3 or x.dim() == 4)
    for pair in flipRef:
        dim0, dim1 = pair
        dim0 -= 1
        dim1 -= 1
        if x.dim() == 4:
            tmp = x[:, dim1].clone()
            x[:, dim1] = x[:, dim0].clone()
            x[:, dim0] = tmp.clone()
            #x[:, dim0], x[:, dim1] = deepcopy((x[:, dim1], x[:, dim0]))
        else:
            tmp = x[dim1].clone()
            x[dim1] = x[dim0].clone()
            x[dim0] = tmp.clone()
            #x[dim0], x[dim1] = deepcopy((x[dim1], x[dim0]))
    return x


def flip(x):
    assert (x.dim() == 3 or x.dim() == 4)
    if '0.4.1' in torch.__version__:
        dim = x.dim() - 1

        return x.flip(dims=(dim,))
    else:
        is_cuda = False
        if x.is_cuda:
            x = x.cpu()
            is_cuda = True
        x = x.numpy().copy()
        if x.ndim == 3:
            x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
        elif x.ndim == 4:
            for i in range(x.shape[0]):
                x[i] = np.transpose(
                    np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
        x = torch.from_numpy(x.copy())
        if is_cuda:
            x = x.cuda()
    return x

# if __name__ == '__main__':
#     img = load_image('/media/ubuntu/文档/code/Pose/img/a.jpg')
#     # rl = torch.Tensor((10, 10))
#     # tb = torch.Tensor((300, 600))
#
#     # s = cropBox(img, rl, tb, 56, 56)
#     # s = torch_to_im(s)
#
#     img = cv_rotate(img, 0.3, 56, 64)
#     img = torch_to_im(img)
#     plt.imshow(img)
#     plt.waitforbuttonpress(0)