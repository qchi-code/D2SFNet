"""
Utils for Dataset
Extended from ADNet code by Hansen et al.
"""
import logging
import random

import cv2
import numpy as np
import torch


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def apply_dct_on_blocks(image, block_size=8):
    height, width = image.shape

    blocks_v = height // block_size
    blocks_h = width // block_size

    dct_blocks = np.zeros((blocks_v, blocks_h, block_size, block_size), dtype=np.float32)

    for i in range(blocks_v):
        for j in range(blocks_h):
            block = image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            dct_block = cv2.dct(np.float32(block))
            dct_blocks[i, j] = dct_block
    flattened_dct_blocks = dct_blocks.reshape(blocks_v, blocks_h, -1)
    flattened_dct_blocks = np.moveaxis(flattened_dct_blocks, 2, 0)

    return flattened_dct_blocks


import torchvision.transforms.functional as F
from PIL import Image
import pywt


def add_gaussian_noise(image, std=0.2):
    """
    给图像添加高斯噪声
    image: torch tensor [1,3,H,W]，归一化到[0,1]
    """
    noise = torch.randn_like(image) * std
    # noisy_image = torch.clamp(image + noise, 0.0, 1.0)
    noisy_image = image + noise
    return noisy_image


def random_spatial_transform(image, mask, degrees=30, scale_range=(0.8, 1.2)):
    angle = random.uniform(-degrees, degrees)
    scale = random.uniform(*scale_range)
    translate = [random.randint(-10, 10), random.randint(-10, 10)]

    # 转为 PIL 图像
    img_pil = F.to_pil_image(image.squeeze(0))
    mask_pil = Image.fromarray(mask.squeeze(0).cpu().numpy().astype(np.uint8), mode='L')

    # 应用仿射变换
    img_pil = F.affine(img_pil, angle=angle, translate=translate, scale=scale, shear=0)
    mask_pil = F.affine(mask_pil, angle=angle, translate=translate, scale=scale, shear=0)

    # 转回 tensor
    img_tensor = F.to_tensor(img_pil).unsqueeze(0)  # [1,3,H,W]
    mask_tensor = torch.from_numpy(np.array(mask_pil)).long().unsqueeze(0)  # [1,H,W]
    return img_tensor, mask_tensor


def dct2(block):
    """对图像块应用二维DCT"""
    return cv2.dct(np.float32(block))


def apply_dct_to_tensor(tensor, block_size=8):
    """
    对tensor中的每个图像应用DCT变换
    参数：
        tensor: torch.Tensor, 形状为 [N, 1, H, W]
        block_size: int, DCT块大小 (建议 4, 8, 或 16)
    返回：
        torch.Tensor, DCT系数张量，形状 [N, block_size*block_size, H//block_size, W//block_size]
    """
    if tensor.size(2) % block_size != 0 or tensor.size(3) % block_size != 0:
        raise ValueError(f"图像尺寸必须是{block_size}的倍数")

    N, _, H, W = tensor.size()
    dct_coeffs = np.zeros((N, H // block_size, W // block_size, block_size, block_size), dtype=np.float32)

    tensor_np = tensor.cpu().numpy().squeeze(1)  # 转为numpy去掉通道维度

    for i in range(N):
        image = tensor_np[i]
        for y in range(0, H, block_size):
            for x in range(0, W, block_size):
                block = image[y:y + block_size, x:x + block_size]
                dct_coeffs[i, y // block_size, x // block_size] = dct2(block)

    # 转换回torch张量
    dct_coeffs_tensor = torch.from_numpy(dct_coeffs)
    dct_coeffs_tensor = dct_coeffs_tensor.flatten(3)  # (N, H//b, W//b, b*b)
    dct_coeffs_tensor = dct_coeffs_tensor.permute(0, 3, 1, 2)  # (N, b*b, H//b, W//b)

    return dct_coeffs_tensor


def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CLASS_LABELS = {
    'CHAOST2': {
        'pa_all': set(range(1, 5)),
        0: set([1, 4]),  # upper_abdomen, leaving kidneies as testing classes
        1: set([2, 3]),  # lower_abdomen
    },
}


def get_bbox(fg_mask, inst_mask):
    """
    Get the ground truth bounding boxes
    """

    fg_bbox = torch.zeros_like(fg_mask, device=fg_mask.device)
    bg_bbox = torch.ones_like(fg_mask, device=fg_mask.device)

    inst_mask[fg_mask == 0] = 0
    area = torch.bincount(inst_mask.view(-1))
    cls_id = area[1:].argmax() + 1
    cls_ids = np.unique(inst_mask)[1:]

    mask_idx = np.where(inst_mask[0] == cls_id)
    y_min = mask_idx[0].min()
    y_max = mask_idx[0].max()
    x_min = mask_idx[1].min()
    x_max = mask_idx[1].max()
    fg_bbox[0, y_min:y_max + 1, x_min:x_max + 1] = 1

    for i in cls_ids:
        mask_idx = np.where(inst_mask[0] == i)
        y_min = max(mask_idx[0].min(), 0)
        y_max = min(mask_idx[0].max(), fg_mask.shape[1] - 1)
        x_min = max(mask_idx[1].min(), 0)
        x_max = min(mask_idx[1].max(), fg_mask.shape[2] - 1)
        bg_bbox[0, y_min:y_max + 1, x_min:x_max + 1] = 0
    return fg_bbox, bg_bbox


def t2n(img_t):
    """
    torch to numpy regardless of whether tensor is on gpu or memory
    """
    if img_t.is_cuda:
        return img_t.data.cpu().numpy()
    else:
        return img_t.data.numpy()


def to01(x_np):
    """
    normalize a numpy to 0-1 for visualize
    """
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-5)


class Scores():

    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.patient_dice = []
        self.patient_iou = []

    def record(self, preds, label):
        assert len(torch.unique(preds)) < 3

        tp = torch.sum((label == 1) * (preds == 1))
        tn = torch.sum((label == 0) * (preds == 0))
        fp = torch.sum((label == 0) * (preds == 1))
        fn = torch.sum((label == 1) * (preds == 0))

        self.patient_dice.append(2 * tp / (2 * tp + fp + fn))
        self.patient_iou.append(tp / (tp + fp + fn))

        self.TP += tp
        self.TN += tn
        self.FP += fp
        self.FN += fn

    def compute_dice(self):
        return 2 * self.TP / (2 * self.TP + self.FP + self.FN)

    def compute_iou(self):
        return self.TP / (self.TP + self.FP + self.FN)


def set_logger(path):
    logger = logging.getLogger()
    logger.handlers = []
    formatter = logging.Formatter('[%(levelname)] - %(name)s - %(message)s')
    logger.setLevel("INFO")

    # log to .txt
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # log to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def low_freq_mutate_np(amp_src, amp_trg, L=0.1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    a_src[:, h1:h2, w1:w2] = a_trg[:, h1:h2, w1:w2]
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


def FDA_source_to_target_np(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img.cpu().detach().numpy()
    trg_img_np = trg_img.cpu().detach().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))
    fft_trg_np = np.fft.fft2(trg_img_np, axes=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


def wavelet_source_to_target_np(src_img, trg_img, level=1):
    src_img_np = src_img.cpu().detach().numpy()
    trg_img_np = trg_img

    coeffs_src = pywt.wavedec2(src_img_np, 'haar', level=level)
    coeffs_trg = pywt.wavedec2(trg_img_np, 'haar', level=level)

    coeffs_src[0] = coeffs_trg[0]

    # 重构图像
    src_in_trg = pywt.waverec2(coeffs_src, 'haar')

    return src_in_trg
