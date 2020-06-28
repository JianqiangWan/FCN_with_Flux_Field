from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
import torch
from PIL import Image
from vis_flux import vis_flux, label2color
import pandas as pd
import cv2
import matplotlib
matplotlib.use('agg')
import pylab as plt
import math
from matplotlib import cm

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_dir:
        os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    step = 0

    calculate_matrix = np.zeros((21, 11))

    for idx in dataset:
        net.forward()
        step += 1
        print(step)

        pred_logits = net.blobs[layer].data[0]
        pred_logits = torch.from_numpy(pred_logits)
        pred_probs = torch.softmax(pred_logits, dim=0).numpy()

        max_value = np.max(pred_probs, axis=0)
        pred_seg = pred_probs.argmax(0)
        
        pred_classes = np.unique(pred_seg)

        gt_seg = net.blobs[gt].data[0, 0]

        pred_flux = net.blobs['flux_score'].data[0]
        norm_pred = np.sqrt(pred_flux[1,:,:]**2 + pred_flux[0,:,:]**2)
        angle_pred = 180/math.pi*np.arctan2(pred_flux[1,:,:], pred_flux[0,:,:])


        # cv2.imwrite(str(idx) + '.png', label2color(gt_seg))

        h, w = pred_seg.shape

        fig = plt.figure(figsize=(14,10))

        ax1 = fig.add_subplot(231)
        ax1.set_title('gt_seg')
        ax1.imshow(label2color(gt_seg)[:,:,::-1])

        ax2 = fig.add_subplot(232)
        ax2.set_title('pred_seg')
        ax2.imshow(label2color(pred_seg)[:,:,::-1])

        ax3 = fig.add_subplot(233)
        ax3.set_title('pred_prob')
        ax3.set_autoscale_on(True)
        im3 = ax3.imshow(max_value, cmap=cm.jet)
        plt.colorbar(im3,shrink=0.5)

        ax1 = fig.add_subplot(234)
        ax1.set_title('Norm_pred')
        ax1.set_autoscale_on(True)
        im1 = ax1.imshow(norm_pred, cmap=cm.jet)
        plt.colorbar(im1,shrink=0.5)

        ax2 = fig.add_subplot(235)
        ax2.set_title('Angle_pred')
        ax2.set_autoscale_on(True)
        im2 = ax2.imshow(angle_pred, cmap=cm.jet)
        plt.colorbar(im2, shrink=0.5)


        plt.savefig('val_prob/' + str(idx) + '.png')
        plt.close(fig)


        '''
        for pred_class in pred_classes:

            tmp_seg = (pred_seg == pred_class)
            calculate_matrix[pred_class, -1] += tmp_seg.sum()

            calculate_matrix[pred_class, 0] += (tmp_seg & (gt_seg == pred_class) & (max_value >= 0.5)).sum()
            calculate_matrix[pred_class, 1] += (tmp_seg & (gt_seg == pred_class) & (max_value >= 0.6)).sum()
            calculate_matrix[pred_class, 2] += (tmp_seg & (gt_seg == pred_class) & (max_value >= 0.7)).sum()
            calculate_matrix[pred_class, 3] += (tmp_seg & (gt_seg == pred_class) & (max_value >= 0.8)).sum()
            calculate_matrix[pred_class, 4] += (tmp_seg & (gt_seg == pred_class) & (max_value >= 0.9)).sum()

            calculate_matrix[pred_class, 5] += (tmp_seg & (max_value >= 0.5)).sum()
            calculate_matrix[pred_class, 6] += (tmp_seg & (max_value >= 0.6)).sum()
            calculate_matrix[pred_class, 7] += (tmp_seg & (max_value >= 0.7)).sum()
            calculate_matrix[pred_class, 8] += (tmp_seg & (max_value >= 0.8)).sum()
            calculate_matrix[pred_class, 9] += (tmp_seg & (max_value >= 0.9)).sum()

        '''

    # data = pd.DataFrame(calculate_matrix)
    # data.to_csv('data.csv')

    '''
    hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                            net.blobs[layer].data[0].argmax(0).flatten(),
                            n_cl)
    # if step < 5:
    vis_flux(net.blobs[gt].data[0, 0],  net.blobs[layer].data[0].argmax(0), net.blobs['flux_score'].data[0], 'vis_pred_flux/' + idx + '.png')
    # step += 1
    if save_dir:
        im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
        im.save(os.path.join(save_dir, idx + '.png'))
    # compute the loss as well
    loss += net.blobs['loss'].data.flat[0]
    '''


    return hist, loss / len(dataset)

def seg_tests(solver, save_format, dataset, layer='score', gt='label'):
    print('>>>', datetime.now(), 'Begin seg tests')
    solver.test_nets[0].share_with(solver.net)
    do_seg_tests(solver.test_nets[0], solver.iter, save_format, dataset, layer, gt)

def do_seg_tests(net, iter, save_format, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iter)
    hist, loss = compute_hist(net, save_format, dataset, layer, gt)
    # mean loss
    # print('>>>', datetime.now(), 'Iteration', iter, 'loss', loss)
    # # overall accuracy
    # acc = np.diag(hist).sum() / hist.sum()
    # print('>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc)
    # # per-class accuracy
    # acc = np.diag(hist) / hist.sum(1)
    # print('>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc))
    # # per-class IU
    # iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    # print('>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu))
    # with open('results/{}_{}.txt'.format(iter, np.nanmean(iu)), 'w'):
    #     pass
    # freq = hist.sum(1) / hist.sum()
    # print('>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
    #         (freq[freq > 0] * iu[freq > 0]).sum())

    return hist
