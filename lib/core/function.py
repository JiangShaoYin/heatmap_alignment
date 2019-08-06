# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import pylab
from matplotlib.patches import Circle

import numpy as np
import torch

from core.config import get_model_name
from core.evaluate import accuracy
from core.evaluate import normalisedError
from core.inference import get_final_preds
from utils.transforms import flip_back
from core.inference import get_max_preds
from utils.vis import save_debug_images
import cv2


logger = logging.getLogger(__name__)


def convert(tensor, overlay=False):
    heatmap = tensor.data.cpu().numpy()[0]
    #heatmap = np.clip(heatmap, 0.0, 1.0)
    if overlay:
        heatmap = np.sum(heatmap, axis=0)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    heatmap = (heatmap*255).astype(np.uint8)
    return heatmap


def visualize(input, output):
    face = convert(input).transpose([1, 2, 0])
    heatmap = convert(output, overlay=True)
    heatmap = cv2.resize(heatmap, (256, 256))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    canvas = face.copy()
    canvas = heatmap*0.7+canvas*0.3
    dispimg = np.hstack([face, canvas]).astype(np.uint8)
    return dispimg


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir):
    # , writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader): # target是label里面的landmark的heatmap，每个点一张，该点为1，其他点为0
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input)  # input为输入的1个batch的图片(经过归一化后的)，32 * 3 *256 * 256, 输出32 * 5 * 64 * 64的图像

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        loss = criterion(output, target, target_weight) # criterion是个JointsMSELoss对象，返回各个特征点误差的欧式距离的平均值

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

       # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())

        # if config.TEST.FLIP_TEST:
            # this part is ugly, because pytorch has not supported negative index
            # input_flipped = model(input[:, :, :, ::-1])
        # input_flipped = np.flip(input.cpu().numpy(), 3).copy()
        # input_flipped = torch.from_numpy(input_flipped).cuda()
        # output_flipped = model(input_flipped)
        # output_flipped = flip_back(output_flipped.cpu().numpy(),
        #                            train_loader.dataset.flip_pairs)
        # output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

        # # feature is not aligned, shift flipped heatmap for higher accuracy
        # if config.TEST.SHIFT_HEATMAP:
        #     output_flipped[:, :, :, 1:] = \
        #         output_flipped.clone()[:, :, :, 0:-1]
        #     # output_flipped[:, :, :, 0] = 0

        # output = (output + output_flipped) * 0.5

        # batch_error_mean = normalisedError(landmark,output)


        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:     # 输出打印信息
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)
            # print("batch_error_mean:{}".format(batch_error_mean))

            # writer = writer_dict['writer']
            # global_steps = writer_dict['train_global_steps']
            # writer.add_scalar('train_loss', losses.val, global_steps)
            # writer.add_scalar('train_acc', acc.val, global_steps)
            # writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            #save_debug_images(config, input, meta, target, pred*4, output,prefix)

        if i%5 == 0:
            dispimg = visualize(input, output)
            # cv2.imshow("result", dispimg)
            if cv2.waitKey(1) == 27:
                exit()

def validate(config, val_loader, val_dataset, model, criterion, output_dir,tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    model.eval()
    total_error = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):                 # compute output
            output = model(input)
            output_256 = model(meta['img_resize256_BN'])

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()  # 计算翻转图像的预测结果
                input_flipped = torch.from_numpy(input_flipped).cuda()
                output_flipped = model(input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]  # 将结果向右偏移1个单位
                    # output_flipped[:, :, :, 0] = 0

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)
            # 将heatmap的点，变成特征点上的图。
            num_images = input.size(0)

            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())
            acc.update(avg_acc, cnt)

            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()  # 因为图像从Dataset里面出来的时候，加了个随机偏移，所以在测试的时候，要把图像偏回去
            s = meta['scale'].numpy()

            # 比较pred和gt的heatmap值
            pred_heatmap,_ = get_max_preds(output.clone().cpu().numpy())
            target_heatmap, _ = get_max_preds(target.clone().cpu().numpy())

            pred_heatmap_256, _ = get_max_preds(output_256.clone().cpu().numpy())
            pred_heatmap_256 *= 4

            preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c, s)  # 变回到250那个尺度上

            gt_landmark = meta['joints'].numpy()
            imgs_256 = meta["img_256"]
            # for img_idx in range(imgs_256.shape[0]):
            #     # vis_face(imgs_256[img_idx], gt_landmark[img_idx], str(img_idx) + ".jpg")
            #     # vis_face(imgs_256[img_idx], preds[img_idx], str(img_idx) + ".jpg")
            #     vis_face(meta['img_resize256'][img_idx], pred_heatmap_256[img_idx], str(img_idx) + ".jpg")

            # batch_error_mean = normalisedError(gt_landmark, preds)
            batch_error_mean = normalisedError(gt_landmark, preds)
            total_error += batch_error_mean
            total_mean_error = total_error / (i+1)
            print("batch id:{0}, current batch mean error is:{1}, total mean error is:{2}".format(i, batch_error_mean,total_mean_error))



# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )
def vis_face(im_array, landmark, save_name):
    pylab.imshow(im_array)  # im_array是输入图片

    if landmark is not None:

        for j in range(5):                                # 遍历5个特征点的坐标
            cir1 = Circle(xy=(landmark[j, 0], landmark[j, 1]), radius=2, alpha=0.4, color="red")
            pylab.gca().add_patch(cir1)

        pylab.savefig(save_name)
        pylab.show()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
