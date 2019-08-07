# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

import random
# from kscv import imgproc

logger = logging.getLogger(__name__)
import skimage

JPEG_FLAG = int(cv2.IMWRITE_JPEG_QUALITY)
def gaussian_sharpen(img, factor=0.5):
    blur_img = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, factor+1, blur_img, -factor, 0)
    return img
def add_jpeg_noise(img, quality=20):
    jpeg_data = cv2.imencode(
                    '.jpg',
                    img,
                    [JPEG_FLAG, quality])[1]
    restore_img = cv2.imdecode(
                    np.array(jpeg_data),
                    cv2.IMREAD_UNCHANGED)
    return restore_img
def add_upscale_noise(img, scale_factor=2, iter_num=1):
    img_h, img_w = img.shape[:2]

    scale_h = int(img_h/scale_factor)
    scale_w = int(img_w/scale_factor)

    for i in range(iter_num):
        scale_img = cv2.resize(img,
                        (scale_w, scale_h),
                        0,
                        0,
                        cv2.INTER_NEAREST)
        img = cv2.resize(scale_img,
                        (img_w, img_h),
                        0,
                        0,
                        cv2.INTER_NEAREST)
    return img
def add_gaussian_noise(img, var):
    noise_img = skimage.util.random_noise(
                        img,
                        mode='gaussian',
                        mean=0,
                        var=var,
                        seed=None,
                        clip=True
                        )
    noise_img = (noise_img*255).astype(np.uint8)
    return noise_img



def augmentation(img):
    if random.uniform(0, 1)<0.5:
        jpeg_noise = random.randint(17, 25)
        # img = imgproc.add_jpeg_noise(img, quality=jpeg_noise)
        img = add_jpeg_noise(img, quality=jpeg_noise)
    
    if random.uniform(0, 1)<0.3:
        scale_factor = random.uniform(1.5, 2.5)
        # img = imgproc.add_upscale_noise(img, scale_factor=scale_factor)
        img = add_upscale_noise(img, scale_factor=scale_factor)
    if random.uniform(0, 1)<0.3:
        gaussian_factor = random.uniform(0, 0.01)
        # img = imgproc.add_gaussian_noise(img, var=gaussian_factor)
        img = add_gaussian_noise(img, var=gaussian_factor)

    return img




class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP

        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        self.sigma = cfg.MODEL.EXTRA.SIGMA

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])  # img，及5个特征点坐标
        image_file = db_rec['image'] # img path
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # 原始图片的尺寸
        cv2.imwrite("data_numpy.jpg", data_numpy)
        img_raw = copy.deepcopy(data_numpy)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints']
        joints_raw = copy.deepcopy(db_rec['joints'])
        joints_vis = db_rec['joints_vis']

        img_resize256 = cv2.resize(data_numpy, (256, 256))  # resize成256， 直接写入meta，返回输出

        joints_256 = np.array([[0 for i in range(2)] for j in range(5)])
        joints_256[:,0] = joints[:,0] * 256 / img_raw.shape[0]
        joints_256[:,1] = joints[:, 1] * 256 / img_raw.shape[0]

        target_256_64, target_weight = self.generate_target(joints_256, joints_vis)  # 对进行仿射变换后的label，生成heatmap


        data_numpy = cv2.resize(data_numpy,(250, 250))

        joints[:,0] = joints[:,0] * 250 / img_raw.shape[0]  # 将label中的特征点，缩放到250这个级别
        joints[:,1] = joints[:,1] * 250 / img_raw.shape[1]

        # drift
        c = np.array([125.0 + random.uniform(-30.0, 30.0),
                      125.0 + random.uniform(-30.0, 30.0)])  # db_rec['center'], 中心点，偏移后的量

        s = 1.0  # db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:  # 训练时，做缩放和旋转，测试时不做
            sf = self.scale_factor # 缩放因子
            rf = self.rotation_factor # 旋转因子
            #s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            s = s * np.clip(np.random.randn()*sf + 1, 0.7, 1.2)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size) # 定义trans， img做缩放，平移和翻转，将图片扩充为256, trans比例为rand（以240为例）-256
        input = cv2.warpAffine(                                # 对input做放射变换到256
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        img_256 = copy.deepcopy(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)  # 对label做放射变换到256

        input = augmentation(input)  # 图像加噪音

        if self.transform:
            input = self.transform(input)  # 对input做transform，为tensor（除以255），再做BN
        if self.transform:
            img_resize256_BN = self.transform(img_resize256)  # 对input做transform，为tensor（除以255），再做BN

        target, target_weight = self.generate_target(joints, joints_vis) # 对进行仿射变换后的label，生成heatmap
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,  # 文件的名字
            'img_raw':img_raw,    # img的pixel数组
            'img_resize256':img_resize256,
            'img_resize256_BN':img_resize256_BN,
            'img_256':img_256,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_raw':joints_raw,  # txt中的label信息
            'joints_256':joints_256,
            'target_256_64':target_256_64,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }
        # return input, target, target_weight  # targer_weight用于控制特征点显示不显示
        return input, target, target_weight, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):  # 遍历5个特征点
                feat_stride = self.image_size / self.heatmap_size # feat_stride[0]，feat_stride[1]分别表示在高， 宽方向的步长
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)  # 图像由256 * 256转换成 64 * 64后，特征点idx的新坐标mu_x =43
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)  # 图像由256 * 256转换成 64 * 64后，特征点idx的新坐标mu_y =23
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]              # ul == up left（x,y）
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]      # br == bottom right(x,y)
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:                             # 处理越界
                    # If not, just return the image as is
                    target_weight[joint_id] = 0 # 将越界的点，weight置为0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)  # x = [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12.]
                y = x[:, np.newaxis]  # y = [[ 0.] [ 1.] [ 2.] [ 3.][ 4.] [ 5.] [ 6.] [ 7.] [ 8.] [ 9.] [10.] [11.] [12.]]
                x0 = y0 = size // 2 # x0 = y0 == 6, 中心点坐标
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))  # 高斯核， 中心点值最大，为1

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])  # heatmap上x的范围，img_x = (37, 50)
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])  # heatmap上y的范围，(17, 30)

                v = target_weight[joint_id] # joint_id 为 1,2,3,4,5
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]  # 将高斯核的值赋上去

        return target, target_weight



