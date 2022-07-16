from __future__ import absolute_import, division, print_function

import math
import os
import random
import PIL
from PIL import Image as pil
import matplotlib.pyplot as PLT
import cv2

import numpy as np

import shutil
import torch
import torch.utils.data as data
from scipy.ndimage.filters import gaussian_filter

from torchvision import transforms
from collections import namedtuple
from .mono_dataset import MonoDataset
class Nuscenes(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(Nuscenes, self).__init__(*args, **kwargs)
        self.root_dir = "./data/Nuscenes"

    def get_image_path(self, root_dir, frame_index):
        file_name = frame_index.replace(
            "both_bev_gt", "trainval").replace(
            "png", "jpg")
        img_path = os.path.join(root_dir, file_name)
        return img_path
    def get_color(self, root_dir, frame_index, do_flip):
        color = self.loader(self.get_image_path(root_dir, frame_index))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    def get_both_path(self, root_dir, frame_index):
        fname = "both_gt_label"
        file_name = frame_index.replace(
            "road_gt", fname)
        path = os.path.join(root_dir, file_name)
        return path
    def get_both(self, root_dir, frame_index, do_flip):
        both = self.loader(self.get_both_path(root_dir, frame_index))
        if do_flip:
            both = both.transpose(pil.FLIP_LEFT_RIGHT)

        return both
    def get_static_path(self, root_dir, frame_index):
        path = os.path.join(root_dir, frame_index)
        return path

    def get_osm_path(self, root_dir):
        osm_file = np.random.choice(os.listdir(root_dir))
        osm_path = os.path.join(root_dir, osm_file)

        return osm_path

    def get_dynamic_path(self, root_dir, frame_index):
        fname = self.opt.seg_class + "_bev_gt"
        file_name = frame_index.replace(
            "road_gt", fname).replace(
            "png", "jpg")
        path = os.path.join(root_dir, file_name)
        return path



    def get_static_gt_path(self, root_dir, frame_index):
        path = os.path.join(
            root_dir,
            frame_index).replace(
            "road_bev",
            "road_gt")
        return path

    def get_dynamic_gt_path(self, root_dir, frame_index):
        return self.get_dynamic_path(root_dir, frame_index)