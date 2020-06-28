import caffe
import numpy as np
import cv2
import random
import torch


class WeightedEuclideanLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check inputs
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute loss.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # def dist and weight for backpropagation
        self.distL1 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.distL2 = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.weight = torch.from_numpy(bottom[2].data)
        # L1 and L2 distance
        self.distL1 = bottom[0].data - bottom[1].data
        self.distL2 = self.distL1**2

        pred_flux = torch.from_numpy(bottom[0].data)
        pred_flux.requires_grad = True

        gt_flux = torch.from_numpy(bottom[1].data)
        gt_flux_normalize = 0.999999 * gt_flux / (gt_flux.norm(p=2, dim=1) + 1e-9)

        loss = torch.sum(self.weight * (pred_flux - gt_flux_normalize)**2)
        loss *= 100
        loss.backward()

        self.torch_grad = pred_flux.grad.numpy()

        top[0].data[...] = loss.item()

        
    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = self.torch_grad
        bottom[1].diff[...] = 0
        bottom[2].diff[...] = 0

class WeightedAngleLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check inputs
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute loss.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # compute angle loss

        self.weight = torch.from_numpy(bottom[2].data)

        pred_direction_field = torch.from_numpy(bottom[0].data)
        pred_direction_field.requires_grad = True
        
        gt_direction_field = torch.from_numpy(bottom[1].data)

        gt_direction_field_normalize = 0.999999 * gt_direction_field / (gt_direction_field.norm(p=2, dim=1) + 1e-9)
        pred_direction_field_normalize = 0.999999 * pred_direction_field / (pred_direction_field.norm(p=2, dim=1) + 1e-9)

        angle_loss = self.weight * (torch.acos(torch.sum(pred_direction_field_normalize * gt_direction_field_normalize, dim=1)))**2
        angle_loss = angle_loss.sum()
        angle_loss *= 20

        angle_loss.backward()

        self.input_grad = pred_direction_field.grad.numpy()
        
        top[0].data[...] = angle_loss.item()

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = self.input_grad
        bottom[1].diff[...] = 0
        bottom[2].diff[...] = 0