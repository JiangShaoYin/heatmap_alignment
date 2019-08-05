# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging
import os
import json_tricks as json

import numpy as np
from scipy.io import loadmat, savemat

from dataset.JointsDataset import JointsDataset


logger = logging.getLogger(__name__)


class MPIIDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, train_set, transform=None):
        super(MPIIDataset, self).__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 5
        self.flip_pairs = [[1, 2], [3, 4]]
        self.parent_ids = [0, 0, 0, 0, 0]
        if train_set:
            self.db = self._get_db()
        else:
            self.db = self._get_db_lfw5590()  #

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # create train/val split
        annot_root = "../data/annots/" # 读取根目录
        #annot_root = "/home/chengchao/tmp/"
        
        gt_db = []
        annots = []

        # for dataset in ["celeba.txt", "106data.txt"]:
        for dataset in ["celeba.txt"]:
            f = open(os.path.join(annot_root, dataset), "r")
            annots.extend(f.readlines())

        for annot in annots:
            words = annot.strip().split()
            image_path = words[0]  #os.path.join(root, words[0])

            joints = np.zeros((self.num_joints, 2), dtype=np.float)
            joints_vis = np.zeros((self.num_joints,  1), dtype=np.float)

            pt_vals = [float(v)*250 for v in words[1:]]
            for i in range(5):
                joints[i, :] = float(pt_vals[2*i]), float(pt_vals[2*i+1])
                joints_vis[:, 0] = 1

            gt_db.append({
                'image': image_path, 
                'joints': joints,
                'joints_vis': joints_vis,
                'filename': '',
                'imgnum': 0,
                })

        return gt_db

    def _get_db_lfw5590(self):
        # create train/val split
        annot_root = "../../mtcnn-pytorch/data_set/face_landmark/"  # 读取根目录
        # annot_root = "/home/chengchao/tmp/"
        print()
        gt_db = []
        annots = []

        for dataset in ["1200.txt"]:  # ["0.txt"]:
            print(os.path.abspath(".")) # 返回参数的绝对路径（string）
            print(os.path.abspath("../../"))
            f = open(os.path.join(annot_root, dataset), "r")
            annots.extend(f.readlines())

        for annot in annots:
            words = annot.strip().split()
            image_path = words[0]  # os.path.join(root, words[0])
            image_path = image_path.replace("\\", "/")
            image_path = image_path.replace("./", "/")
            image_path = "../../mtcnn-pytorch" + image_path


            joints = np.zeros((self.num_joints, 2), dtype=np.float)
            joints_vis = np.zeros((self.num_joints, 1), dtype=np.float)

            pt_vals = [v for v in words[6:]]
            for i in range(5):
                joints[i, :] = float(pt_vals[2 * i]), float(pt_vals[2 * i + 1])
                joints_vis[:, 0] = 1
            joints[0,:], joints[2,:] = joints[2,:].copy(), joints[0,:].copy()
            joints[1, :], joints[2, :] = joints[2, :].copy(), joints[1, :].copy()


            gt_db.append({
                'image': image_path,
                'joints': joints,
                'joints_vis': joints_vis,
                'filename': '',
                'imgnum': 0,
            })

        return gt_db

