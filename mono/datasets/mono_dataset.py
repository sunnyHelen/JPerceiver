#
from __future__ import absolute_import, division, print_function
import random
import math
import os
import numpy as np
from PIL import Image as pil # using pillow-simd for increased speed
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.utils.data as data
from torchvision import transforms
from collections import namedtuple
# import numpy as np
import torch
import torch.utils.data as data
import pykitti
from scipy.ndimage.filters import gaussian_filter
import linecache
import warnings
warnings.filterwarnings('ignore')
def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 type,
                 is_train=False,
                 img_ext='.png',
                 gt_depth_path=None
                 ):
        super(MonoDataset, self).__init__()
        self.interp = Image.ANTIALIAS
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.img_ext = img_ext
        self.loader = pil_loader
        self.gt_depth_path = gt_depth_path
        self.to_tensor = transforms.ToTensor()
        self.type = type

        # Need to specify augmentations differently in pytorch 1.0 compared with 0.4
        if int(torch.__version__.split('.')[0]) > 0:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
        else:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = transforms.Resize((self.height, self.width), interpolation=self.interp)
        if self.type == "static" or self.type == "dynamic" or self.type == "static_raw":
            self.resize_full = transforms.Resize((375, 1242), interpolation=self.interp)
            self.K = np.array([[0.58, 0, 0.5, 0],
                               [0, 1.92, 0.5, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=np.float32)

            self.full_res_shape = (1242, 375)
        elif self.type == "Argo_static" or self.type == "Argo_dynamic" or self.type == "Argo_both":
            
            self.resize_full = transforms.Resize((2056, 2464), interpolation=self.interp)
        # self.resize_1024 = transforms.Resize((1024, 1024), interpolation=self.interp)

        self.flag = np.zeros(self.__len__(), dtype=np.int64)

        if not is_train and self.gt_depth_path is not None:
            self.gt_depths = np.load(gt_depth_path,
                                     allow_pickle=True,
                                     fix_imports=True, encoding='latin1')["data"]
    def process_K(self, inputs):
        K = self.K.copy()
        K_1 = self.K.copy()
        K_1[0, :] *= self.full_res_shape[0]
        K_1[1, :] *= self.full_res_shape[1]
        inputs[("K", -1)] = torch.from_numpy(K_1)
        # K_1.dropna(inplace=True)
        inv_K_1 = np.linalg.pinv(K_1)
        inputs[("inv_K", -1)] = torch.from_numpy(inv_K_1)
        K[0, :] *= self.width
        K[1, :] *= self.height
        inv_K = np.linalg.pinv(K)

        inputs[("K", 0)] = torch.from_numpy(K)
        # print("K shape----------------", inputs[("K")].shape)
        inputs[("inv_K", 0)] = torch.from_numpy(inv_K)
    def process_K_argo(self, inputs):
        K = inputs[("odometry_K", 0, 0)].copy()
        K[0, :] *= (self.width / self.full_res_shape[0])
        K[1, :] *= (self.height / self.full_res_shape[1])
        inv_K = np.linalg.pinv(K)
        #
        inputs[("K", 0)] = torch.from_numpy(K)
        # # print("K shape----------------", inputs[("K")].shape)
        inputs[("inv_K", 0)] = torch.from_numpy(inv_K)
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        if self.is_train:
            inputs[("color", 0, -1)] = self.resize_full(inputs[("color", 0, -1)])
            inputs[("color", -1, -1)] = self.resize_full(inputs[("color", -1, -1)])
            inputs[("color", 1, -1)] = self.resize_full(inputs[("color", 1, -1)])
        else:
            inputs[("color", 0, -1)] = self.resize_full(inputs[("color", 0, -1)])

        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                # print(n,im,i)
                inputs[(n, im, 0)] = self.resize(inputs[(n, im, - 1)])




        for k in list(inputs):
            if "color" in k:
                f = inputs[k]
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                if i == 0:
                    # print("aug type", type(color_aug))
                    co_aug = color_aug(f)
                    inputs[(n + "_aug", im, i)] = self.to_tensor(co_aug)
        # print("inputs keys*************************", inputs.keys())
        # for key in inputs.keys():

        for key in inputs.keys():
                if key[0] == "both"  and self.type == "both":
                    inputs[key] = torch.from_numpy(inputs[key])
                elif key[0] == "bothS" or key[0] == "bothD" or ( key[0] == "dynamic") and \
                        (self.type == "static" or self.type == "static_raw" or self.type == "dynamic"
                         or self.type == "Argo_static" or self.type == "Argo_dynamic" or self.type == "Argo_both"):
                    inputs[key] = self.process_topview(inputs[key], self.height//4)
                    inputs[key] = self.to_tensor(inputs[key])
                if key[0] == "both_dynamic":
                    inputs[key] = self.process_topview_both(inputs[key], self.height//4)
                    inputs[key] = self.to_tensor(inputs[key])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        frame_index = self.filenames[index]
        # print("%%%%%%%%%%%%%%%%%%",frame_index)
        folder = self.data_path
        if self.type == "static":
            # for KITTI Odometry dataset
            if not self.is_train:
                inputs['gt_depth'] = self.get_depth_odom(folder, frame_index, do_flip)
            odometry_K, Tr_cam2_velo = self.get_static_K(folder, frame_index, 0)
            for i in self.frame_idxs:
                inputs[("odometry_K", i, 0)] = odometry_K
                inputs[("Tr_cam2_velo", i, 0)] = Tr_cam2_velo
                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index, do_flip)
                else:
                    try:
                        inputs[("color", i, -1)] = self.get_color_layout(folder, frame_index, i, do_flip)

                        inputs[("bothS", i, 0)] = self.get_static(self.get_static_path(folder, frame_index, i)[0], do_flip)
                        # inputs[("odometry_K", i, 0)] = self.get_static_K(folder, frame_index, i)
                        # inputs[("Tr_cam2_velo", i, 0)] = self.get_static_Tr_cam2_velo(folder, frame_index, i)
                        # inputs[("bev_path", i, 0)] = {"sequence": int(self.get_static_path(folder, frame_index, i)[1]),
                        #                               "frame_index": int(self.get_static_path(folder, frame_index, i)[2])}
                    except:
                        inputs[("color", i, -1)] = self.get_color_layout(folder, frame_index, 0, do_flip)

                        inputs[("bothS", i, 0)] = self.get_static(self.get_static_path(folder, frame_index, 0)[0], do_flip)
                        # inputs[("odometry_K", i, 0)] = self.get_static_K(folder, frame_index, 0)
                        # inputs[("Tr_cam2_velo", i, 0)] = self.get_static_Tr_cam2_velo(folder, frame_index, 0)
            self.process_K(inputs)
        elif self.type == "static_raw":
            # for KITTI Raw dataset
            if not self.is_train:
                inputs['gt_depth'] = self.get_depth_raw(folder, frame_index, do_flip)
            odometry_K, Tr_cam2_velo = self.get_static_K(folder, frame_index, 0)
            for i in self.frame_idxs:
                inputs[("odometry_K", i, 0)] = odometry_K
                inputs[("Tr_cam2_velo", i, 0)] = Tr_cam2_velo
                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index, do_flip)
                else:
                    try:
                        inputs[("color", i, -1)] = self.get_color_layout(folder, frame_index, i, do_flip)

                        inputs[("bothS", i, 0)] = self.get_static(self.get_static_path(folder, frame_index, i), do_flip)
                        # inputs[("odometry_K", i, 0)] = self.get_static_K(folder, frame_index, i)
                        # inputs[("Tr_cam2_velo", i, 0)] = self.get_static_Tr_cam2_velo(folder, frame_index, i)
                        # inputs[("bev_path", i, 0)] = {"sequence": int(self.get_static_path(folder, frame_index, i)[1]),
                        #                               "frame_index": int(self.get_static_path(folder, frame_index, i)[2])}
                    except:
                        inputs[("color", i, -1)] = self.get_color_layout(folder, frame_index, 0, do_flip)

                        inputs[("bothS", i, 0)] = self.get_static(self.get_static_path(folder, frame_index, 0), do_flip)
                        # inputs[("odometry_K", i, 0)] = self.get_static_K(folder, frame_index, 0)
                        # inputs[("Tr_cam2_velo", i, 0)] = self.get_static_Tr_cam2_velo(folder, frame_index, 0)
                        # inputs[("bev_path", i, 0)] = {"sequence": int(self.get_static_path(folder, frame_index, i)[1]),
                        #                               "frame_index": int(self.get_static_path(folder, frame_index, i)[2])}
            self.process_K(inputs)
        elif self.type == "dynamic":
            # for KITTI Object dataset
            index_file = "/CV/datasets/kitti_data/object/train_randnew.txt"
            mapping_file = "/CV/datasets/kitti_data/object/train_mapping.txt"
            index = linecache.getline(index_file, int(frame_index)+1)
            mapping = linecache.getline(mapping_file, int(index))
            odometry_K, Tr_cam2_velo = self.get_dynamic_K(folder, frame_index, mapping)
            for i in self.frame_idxs:
                inputs[("odometry_K", i, 0)] = odometry_K
                inputs[("Tr_cam2_velo", i, 0)] = Tr_cam2_velo
                try:
                    inputs[("color", i, -1)] = self.get_color_layout(folder, frame_index, i, do_flip)

                    inputs[("bothD", i, 0)] = self.get_dynamic(self.get_dynamic_path(folder, frame_index, i), do_flip)
                except:
                    inputs[("color", i, -1)] = self.get_color_layout(folder, frame_index, 0, do_flip)

                    inputs[("bothD", i, 0)] = self.get_dynamic(self.get_dynamic_path(folder, frame_index, 0), do_flip)
            self.process_K(inputs)
        elif self.type.split('_')[0] == "Argo":
            # for argoverse datset
            if self.is_train:
                frame_index_ = frame_index.split(' ')
                ids = {0: 0, -1: 1, 1: 2}
                frame_index = frame_index_[ids[0]]
            else:
                ids = {0: 0}
                frame_index_ = [frame_index]
            # print("----------------------", frame_index_)
            odometry_K, Tr_cam2_velo = self.get_intrinsic(folder, frame_index)
            if self.type == "Argo_dynamic" or self.type == "Argo_both":
                for i in self.frame_idxs:
                    inputs[("odometry_K", i, 0)], inputs[("Tr_cam2_velo", i, 0)] = odometry_K, Tr_cam2_velo

                    try:
                        frame_index = frame_index_[ids[i]]
                        inputs[("color", i, -1)] = self.get_color(folder, frame_index, i, do_flip)

                        inputs[("bothD", i, 0)] = self.get_dynamic(self.get_dynamic_path(folder, frame_index, i), do_flip)
                    except:
                        frame_index = frame_index_[ids[0]]
                        inputs[("color", i, -1)] = self.get_color(folder, frame_index, 0, do_flip)

                        inputs[("bothD", i, 0)] = self.get_dynamic(self.get_dynamic_path(folder, frame_index, 0), do_flip)
            if self.type == "Argo_static" or self.type == "Argo_both":
                for i in self.frame_idxs:
                    if self.type == "Argo_static":
                        inputs[("odometry_K", i, 0)], inputs[("Tr_cam2_velo", i, 0)] = odometry_K, Tr_cam2_velo
                    try:
                        frame_index = frame_index_[ids[i]]
                        # print(frame_index)
                        if self.type == "Argo_static":
                            inputs[("color", i, -1)] = self.get_color(folder, frame_index, i, do_flip)
                        label_path = self.get_static_path(folder, frame_index, i)
                        inputs[("bothS", i, 0)] = self.get_static(label_path, do_flip)
                        # inputs[("both_dynamic", i, 0)] = self.get_both(folder, frame_index, i, do_flip)
                    except:
                        frame_index = frame_index_[ids[0]]
                        if self.type == "Argo_static":
                            inputs[("color", i, -1)] = self.get_color(folder, frame_index, 0, do_flip)
                        label_path = self.get_static_path(folder, frame_index, 0)
                        inputs[("bothS", i, 0)] = self.get_static(label_path, do_flip)
                        # inputs[("both_dynamic", i, 0)] = self.get_both(folder, frame_index, 0, do_flip)
            if self.type == "Argo_both":
                try:
                    inputs[("both_dynamic", i, 0)] = self.get_both(folder, frame_index, i, do_flip)
                except:
                    inputs[("both_dynamic", i, 0)] = self.get_both(folder, frame_index, 0, do_flip)

            self.process_K_argo(inputs)



        if do_color_aug:
            # color_aug = transforms.ColorJitter.get_params((self.brightness, self.contrast, self.saturation, self.hue))
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        return inputs

    def get_color(self, folder, frame_index, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_pose(self, folder, frame_index, offset):
        return
    def get_both(self, folder, frame_index, do_flip):
        raise NotImplementedError

    def get_intrinsic(self, folder, frame_index, do_flip):
        raise NotImplementedError

    def process_trainId(self, topview, size):#, both_path):

        topview = topview.resize((size, size), Image.NEAREST)

        img = topview.convert("RGB")
        SegClass = namedtuple('SegClass', ['name', 'id', 'train_id', 'color'])
        classes = [
            SegClass('background', 0, 0, (0, 0, 0)),
            SegClass('vehicle', 1, 1, (0, 0, 255)),
            SegClass('road', 2, 2, (255, 255, 255)),]
            #SegClass('pedestrain', 3, 3, (255, 0, 0)), ]
        color2index = {}
        for obj in classes:
            idx = obj.train_id
            label = obj.name
            color = obj.color
            color2index[color] = idx
        img_array = np.array(img)
        # print(img_array.shape)
        height, width, c = img_array.shape

        l = []
        idx_mat = np.zeros((height, width))
        for h in range(height):
            for w in range(width):
                color = tuple(img_array[h, w])
                # print(color)
                # print(color)
                try:
                    index = color2index[color]
                    # if index == 1:
                    #     print(index)
                    idx_mat[h, w] = index
                    if index not in l:
                        l.append(index)
                except:
                    # no index, assign to void
                    idx_mat[h, w] = 0
        idx_mat = idx_mat.astype(np.uint8)
        # im = Image.fromarray(idx_mat)
        # both_path_0 = both_path.rsplit("/", 1)[0]
        # if not os.path.exists(both_path_0):
        #     os.makedirs(both_path_0)
        # if os.path.exists(both_path) and os.path.isdir(both_path):
        #     shutil.rmtree(both_path)
        # # both_path = both_path.replace()
        # if not os.path.exists(both_path):
        #     im.save(both_path)
            # print("saving gt label: ", both_path)
        # print("idx_mat",idx_mat.shape)
        # print(type(idx_mat))
        return idx_mat

    def process_topview(self, topview, size):
        topview = topview.convert("1")
        topview = topview.resize((size, size), pil.NEAREST)
        topview = topview.convert("L")
        topview = np.array(topview)
        topview_n = np.zeros(topview.shape)
        topview_n[topview == 255] = 1  # [1.,0.]
        return topview_n
    def process_topview_both(self, topview, size):
        topview = topview.resize((size, size), pil.NEAREST)

        topview = np.array(topview)
        topview_n = np.zeros(topview.shape)
        topview_n[topview ==255] = 1  # [1.,0.]
        return topview_n
    # def get_both(self, path, do_flip):
    #     try:
    #         tv = self.loader(path)
    #     except PIL.UnidentifiedImageError:
    #         path_static = path.replace( "both_gt", "road_gt")
    #         path_dynamic = path.replace("both_gt", "car_bev_gt").replace("png","jpg")
    #         self.creat_both_gt(self, path_static, path_dynamic, do_flip)
    #         tv = self.loader(path)
    #     if do_flip:
    #         tv = tv.transpose(pil.FLIP_LEFT_RIGHT)
    #     # path = path.replace("both_gt", "both_gt_label")
    #     # idx_mat = process_trainId(tv, path)
    #     return tv#.convert('L')

