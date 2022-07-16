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
import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import cv2
import argoverse
from argoverse.utils.se3 import SE3
from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.utils.calibration import get_calibration_config
class Argoverse(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(Argoverse, self).__init__(*args, **kwargs)
        self.root_dir = "./data/argo"
        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 1.92, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (2464, 2056)
        self.camera_name = 'stereo_front_left'
        self.split_data_dir = None

    def get_image_path(self, root_dir, frame_index, i):
        # print(frame_index)
        split_data_dir = frame_index.split("/")[1]
        self.split_data_dir = f'{root_dir}/argoverse-tracking/{split_data_dir}'
        file_name = f'{root_dir}/{frame_index}'
        self.log_id = frame_index.rsplit("/")[2]
        img_path0 = os.path.join(root_dir, file_name)
        img_path0 = img_path0.replace('road_gt_new', 'stereo_front_left').replace(
            "png", "jpg")
        return img_path0

    def get_color(self, root_dir, frame_index, i, do_flip):
        color = self.loader(self.get_image_path(root_dir, frame_index, i))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    def get_both(self, root_dir, frame_index,i, do_flip):
        both = self.loader(self.get_both_path(root_dir, frame_index, i))
        if do_flip:
            both = both.transpose(pil.FLIP_LEFT_RIGHT)

        return both
    def get_both_path(self, root_dir, frame_index, i):
        path = self.get_image_path(root_dir, frame_index, i)
        path = path.replace('stereo_front_left/', 'both_bev_gt_new/').replace('jpg', 'png')
        return path
    def get_both_gt_path(self, root_dir, frame_index):
        fname ="both_gt_label"
        file_name = frame_index.replace(
            "road_gt_new", fname)
        path = os.path.join(root_dir, file_name)
        return path
    def get_intrinsic(self,root_dir, frame_index):
        #print(frame_index)
        split_data_dir = frame_index.rsplit("/")[1]
        self.split_data_dir = f'{root_dir}/argoverse-tracking/{split_data_dir}'
        self.log_id = frame_index.rsplit("/")[2]
        dl = SimpleArgoverseTrackingDataLoader(data_dir=self.split_data_dir, labels_dir=self.split_data_dir)
        calib_data = dl.get_log_calibration_data(self.log_id)
        camera_config = get_calibration_config(calib_data, self.camera_name)
        K = camera_config.intrinsic#[:,:3]
        K3=[0, 0, 0, 1]
        K = np.vstack([K, K3])
        camera_SE3_egovehicle = camera_config.extrinsic
        # print("------------",K)
        return K, camera_SE3_egovehicle
    # def get_dynamic_Tr_cam2_velo(self, root_dir, frame_index):
    #     split_data_dir = frame_index.rsplit("/")[1]
    #     self.split_data_dir = f'{root_dir}/argoverse-tracking/{split_data_dir}'
    #     self.log_id = frame_index.rsplit("/")[2]
    #     # print("frame_index-----------------", frame_index)
    #     # print("split_data_dir-----------------",self.split_data_dir)
    #     # print("log_id---------------------",self.log_id)
    #     # city = dl.get_city_name(self.log_id)
    #     dl = SimpleArgoverseTrackingDataLoader(data_dir=self.split_data_dir, labels_dir=self.split_data_dir)
    #     calib_data = dl.get_log_calibration_data(self.log_id)
    #     camera_config = get_calibration_config(calib_data, self.camera_name)
    #     camera_SE3_egovehicle = camera_config.extrinsic
    #     return camera_SE3_egovehicle

    def get_static_path(self, root_dir, frame_index, i):
        path = self.get_image_path(root_dir, frame_index, i)
        path = path.replace('stereo_front_left/', 'road_gt_new/').replace('jpg','png')
        return path

    def get_osm_path(self, root_dir):
        osm_file = np.random.choice(os.listdir(root_dir))
        osm_path = os.path.join(root_dir, osm_file)

        return osm_path

    def get_dynamic_path(self, root_dir, frame_index, i):
        path = self.get_image_path(root_dir, frame_index, i)
        path = path.replace('stereo_front_left/','car_bev_gt_new/')
        return path
    def get_dynamic(self, path, do_flip):
        tv = self.loader(path)
        if do_flip:
            tv = tv.transpose(pil.FLIP_LEFT_RIGHT)
        return tv.convert('L')
    def get_static_gt_path(self, root_dir, frame_index):
        path = self.get_image_path(root_dir, frame_index, i)
        path = path.replace('stereo_front_left/', 'road_gt_new/')
        return path
    def get_static(self, path, do_flip):
        tv = self.loader(path)
        if do_flip:
            tv = tv.transpose(pil.FLIP_LEFT_RIGHT)
        return tv.convert('L')
    def get_dynamic_gt_path(self, root_dir, frame_index):
        return self.get_dynamic_path(root_dir, frame_index)
