import sys
import scipy.io as sio
import math
import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import pylab as plt
from matplotlib import cm
import os

def label2color(label):

    label = label.astype(np.uint16)
    
    height, width = label.shape
    color3u = np.zeros((height, width, 3), dtype=np.uint8)
    unique_labels = np.unique(label)

    if unique_labels[-1] >= 2**24:       
        raise RuntimeError('Error: label overflow!')

    for i in range(len(unique_labels)):
    
        binary = '{:024b}'.format(unique_labels[i])
        # r g b 3*8 24
        r = int(binary[::3][::-1], 2)
        g = int(binary[1::3][::-1], 2)
        b = int(binary[2::3][::-1], 2)

        color3u[label == unique_labels[i]] = np.array([r, g, b])

    return color3u


def vis_flux(gt_mask, pred_mask, pred_flux, savedir):


    norm_pred = np.sqrt(pred_flux[1,:,:]**2 + pred_flux[0,:,:]**2)
    angle_pred = 180/math.pi*np.arctan2(pred_flux[1,:,:], pred_flux[0,:,:])

    fig = plt.figure(figsize=(18,6))

    ax1 = fig.add_subplot(141)
    ax1.set_title('Norm_pred')
    ax1.set_autoscale_on(True)
    im1 = ax1.imshow(norm_pred, cmap=cm.jet)
    plt.colorbar(im1,shrink=0.5)

    ax2 = fig.add_subplot(142)
    ax2.set_title('Angle_pred')
    ax2.set_autoscale_on(True)
    im2 = ax2.imshow(angle_pred, cmap=cm.jet)
    plt.colorbar(im2, shrink=0.5)

    ax3 = fig.add_subplot(143)
    ax3.set_title('gt_mask')
    color_mask = label2color(gt_mask)
    ax3.imshow(color_mask)

    ax4 = fig.add_subplot(144)
    ax4.set_title('pred_mask')
    color_mask = label2color(pred_mask)
    ax4.imshow(color_mask)

    plt.savefig(savedir)
    plt.close(fig)
