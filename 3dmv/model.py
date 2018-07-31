
import os, sys, inspect
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

import util
from projection import Projection

# z-y-x coordinates
class Model2d3d(nn.Module):
    def __init__(self, num_classes, num_images, intrinsic, image_dims, grid_dims, depth_min, depth_max, voxel_size):
        super(Model2d3d, self).__init__()
        self.num_classes = num_classes
        self.num_images = num_images
        self.intrinsic = intrinsic
        self.image_dims = image_dims
        self.grid_dims = grid_dims
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.voxel_size = voxel_size
        self.nf0 = 32 
        self.nf1 = 64 
        self.nf2 = 128 
        self.bf = 1024
        column_height = grid_dims[2]
        self.pooling = nn.MaxPool1d(kernel_size=num_images)
        self.features2d = nn.Sequential(
            # output self.nf0 x 30x15x15
            nn.Conv3d(128, 64, kernel_size=[4, 3, 3], stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.Conv3d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.Conv3d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.Dropout3d(0.2),
            # output self.nf1 x 14x7x7
            nn.Conv3d(64, 32, kernel_size=[4, 3, 3], stride=2, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.Conv3d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.Conv3d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.Dropout3d(0.2)
        )
        self.features3d = nn.Sequential(
            # output self.nf0 x 30x15x15
            nn.Conv3d(2, self.nf0, kernel_size=[4, 3, 3], stride=2, padding=0),
            nn.BatchNorm3d(self.nf0),
            nn.ReLU(True),
            nn.Conv3d(self.nf0, self.nf0, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(self.nf0),
            nn.ReLU(True),
            nn.Conv3d(self.nf0, self.nf0, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(self.nf0),
            nn.ReLU(True),
            nn.Dropout3d(0.2),
            # output self.nf1 x 14x7x7
            nn.Conv3d(self.nf0, self.nf1, kernel_size=[4, 3, 3], stride=2, padding=0),
            nn.BatchNorm3d(self.nf1),
            nn.ReLU(True),
            nn.Conv3d(self.nf1, self.nf1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(self.nf1),
            nn.ReLU(True),
            nn.Conv3d(self.nf1, self.nf1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(self.nf1),
            nn.ReLU(True),
            nn.Dropout3d(0.2)
        )
        self.features = nn.Sequential(
            # output self.nf2 x 6x3x3
            nn.Conv3d(self.nf1+32, self.nf2, kernel_size=[4, 3, 3], stride=2, padding=0),
            nn.BatchNorm3d(self.nf2),
            nn.ReLU(True),
            nn.Conv3d(self.nf2, self.nf2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(self.nf2),
            nn.ReLU(True),
            nn.Conv3d(self.nf2, self.nf2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(self.nf2),
            nn.ReLU(True),
            nn.Dropout3d(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.nf2 * 54, self.bf),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.bf, num_classes*column_height)
        )

    def forward(self, volume, image_features, projection_indices_3d, projection_indices_2d, volume_dims):
        assert len(volume.shape) == 5 and len(image_features.shape) == 4
        batch_size = volume.shape[0]
        num_images = projection_indices_3d.shape[0] // batch_size

        # project 2d to 3d
        image_features = [Projection.apply(ft, ind3d, ind2d, volume_dims) for ft, ind3d, ind2d in zip(image_features, projection_indices_3d, projection_indices_2d)]
        image_features = torch.stack(image_features, dim=4)

        # reshape to max pool over features
        sz = image_features.shape
        image_features = image_features.view(sz[0], -1, batch_size * num_images)
        if num_images == self.num_images:
            image_features = self.pooling(image_features)
        else:
            image_features = nn.MaxPool1d(kernel_size=num_images)(image_features)
        image_features = image_features.view(sz[0], sz[1], sz[2], sz[3], batch_size)
        image_features = image_features.permute(4, 0, 1, 2, 3)

        volume = self.features3d(volume)
        image_features = self.features2d(image_features)
        x = torch.cat([volume, image_features], 1)
        x = self.features(x)
        x = x.view(batch_size, self.nf2 * 54)
        x = self.classifier(x)
        x = x.view(batch_size, self.grid_dims[2], self.num_classes)
        return x
