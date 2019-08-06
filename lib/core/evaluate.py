# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0])) # 5 * 32矩阵，表明32个图上，每个特征点与label值之间的距离
    for img_idx in range(preds.shape[0]):
        for landmark_idx in range(preds.shape[1]):
            if target[img_idx, landmark_idx, 0] > 1 and target[img_idx, landmark_idx, 1] > 1:# heatmap中，target里面的landmark的坐标gt值，特征点x，y坐标都大于1
                normed_preds = preds[img_idx, landmark_idx, :] / normalize[img_idx]   # 特征点坐标归一化（ x / 6.4,  y / 6.4）

                normed_targets = target[img_idx, landmark_idx, :] / normalize[img_idx]

                dists[landmark_idx, img_idx] = np.linalg.norm(normed_preds - normed_targets)  # 计算经过normalize后的标准差, 1个特征点之间的距离
            else:
                dists[landmark_idx, img_idx] = -1
    return dists  # 返回32 * 5个特征点之间的两两距离


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal  # 计算dists矩阵32张图中，某一特征点与GT值距离小于0.5的图像个数 / 一个batch中特征点有效的img张数
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1])) # idx为特征点的编号
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10      # 32 * 2上，每个点的值为1 * 64 /10
    dists = calc_dists(pred, target, norm)                              # 在heatmap（64 * 64）上计算预测值和GT值之间的差距，5 * 32维度

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])  # 计算dists矩阵32张图中，heatmap上，某一特征点与GT值距离小于0.5的图像个数 / 一个batch中特征点有效的img张数
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0  # 计算平均误差率
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred

def normalisedError(gt_landmark, preds):
    delta = gt_landmark[:, 2, :] - gt_landmark[:, 3, :]  # pt0是鼻尖，pt1和pt2是左右眼

    normDist = np.linalg.norm(delta, axis=1)  # 左右眼距离
    landmark_error = np.linalg.norm(preds - gt_landmark, axis=2)  # 5个特征点的平方和开根号
    normalised_error = np.sum(landmark_error, axis=1)  # 求多个特征点的总误差
    normalised_error = normalised_error / normDist  # 计算每个批次的误差

    batch_error_mean = np.sum(normalised_error) / normalised_error.shape[0] # 1个batch的平均误差
    return batch_error_mean
