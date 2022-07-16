from __future__ import absolute_import, division, print_function
import sys
import os
import scipy.misc
import numpy as np
import PIL.Image as pil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import datetime
import skimage.transform
from .kitti_utils import generate_depth_map, read_calib_file, transform_from_rot_trans, pose_from_oxts_packet
from .mono_dataset import MonoDataset
import pykitti

class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color



class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path
    def get_color_layout(self, root_dir, frame_index, index, do_flip):
        color = self.loader(self.get_image_path_layout(root_dir, frame_index, index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path_layout(self, root_dir, frame_index, ind):
        index = int(frame_index.split('image_02/data/')[1].split('.')[0])
        img_path = os.path.join(root_dir, frame_index)#.replace(
            # "road_labels", "image_02/data"))
        img_path = img_path.split('/00')[0]
        img_path = os.path.join(img_path, "%010d.png" % int(index + ind))
        return img_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        im = Image.fromarray(depth_gt)
        new_image = np.array(im.resize(size, PIL.Image.BICUBIC))
        depth_gt = scipy.misc.imresize(depth_gt, self.full_res_shape[::-1], "nearest")

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
    def get_depth_raw(self, root_dir, frame_index, do_flip):
        index = int(frame_index.split('image_02/data/')[1].split('.')[0])
        files = frame_index.split('image_02/data/')[0]
        calib_path = os.path.join(root_dir,files.split('/')[0])
        velo_filename = os.path.join(
            root_dir,files,
            "velodyne_points/data/{:010d}.bin".format(int(index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, 2)
        im = pil.fromarray(depth_gt)
        depth_gt = np.array(im.resize(self.full_res_shape[::-1], pil.NEAREST)).astype(np.double)
        # depth_gt = scipy.misc.imresize(depth_gt, self.full_res_shape[::-1], "nearest")

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return np.array(depth_gt.copy())

    def get_pose(self, folder, frame_index, offset):
        oxts_root = os.path.join(self.data_path, folder, 'oxts')
        with open(os.path.join(oxts_root, 'timestamps.txt')) as f:
            timestamps = np.array([datetime.datetime.strptime(ts[:-3], "%Y-%m-%d %H:%M:%S.%f").timestamp()
                                   for ts in f.read().splitlines()])

        speed0 = np.genfromtxt(os.path.join(oxts_root, 'data', '{:010d}.txt'.format(frame_index)))[[8, 9, 10]]
        # speed1 = np.genfromtxt(os.path.join(oxts_root, 'data', '{:010d}.txt'.format(frame_index+offset)))[[8, 9, 10]]

        timestamp0 = timestamps[frame_index]
        timestamp1 = timestamps[frame_index+offset]
        # displacement = 0.5 * (speed0 + speed1) * (timestamp1 - timestamp0)
        displacement = speed0 * (timestamp1 - timestamp0)

        imu2velo = read_calib_file(os.path.join(self.data_path, os.path.dirname(folder), 'calib_imu_to_velo.txt'))
        velo2cam = read_calib_file(os.path.join(self.data_path, os.path.dirname(folder), 'calib_velo_to_cam.txt'))
        cam2cam = read_calib_file(os.path.join(self.data_path, os.path.dirname(folder), 'calib_cam_to_cam.txt'))

        velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])
        imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])
        cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))

        imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat

        odo_pose = imu2cam[:3,:3] @ displacement + imu2cam[:3,3]

        return odo_pose
    def get_both_raw(self, path, path_static, path_dynamic, do_flip):
        self.creat_both_gt(path_static, path_dynamic)
        tv = self.loader(path)
        if do_flip:
            tv = tv.transpose(pil.FLIP_LEFT_RIGHT)
        return tv#.convert('L')
    def get_both_direct(self, path, do_flip):
        tv = self.loader(path)
        if do_flip:
            tv = tv.transpose(pil.FLIP_LEFT_RIGHT)
        tv = np.array(tv)
        tv = tv[:, :, 0]
        return tv
    def get_both_path(self, root_dir, frame_index, ind):
        index = int(frame_index.split('road_labels/')[1].split('.')[0])
        img_path = os.path.join(root_dir, frame_index.replace(
            "road_labels", "both_gt"))
        img_path = img_path.split('/00')[0]
        img_path = os.path.join(img_path, "%010d.png" % int(index + ind))
        return img_path
    def get_both_label_path(self, root_dir, frame_index, ind):
        index = int(frame_index.split('road_labels/')[1].split('.')[0])
        img_path = os.path.join(root_dir, frame_index.replace(
            "road_labels", "both_gt_label"))
        img_path = img_path.split('/00')[0]
        img_path = os.path.join(img_path, "%010d.png" % int(index + ind))
        return img_path
    def get_both_gt_path(self, root_dir, frame_index):
        fname ="both_gt_label"
        file_name = frame_index.replace(
            "road_gt", fname)
        path = os.path.join(root_dir, file_name)
        return path
    def get_intrinsic(self,root_dir, frame_index):

        return self.K

    def get_static(self, path, do_flip):
        tv = self.loader(path)
        if do_flip:
            tv = tv.transpose(pil.FLIP_LEFT_RIGHT)
        return tv.convert('L')
    def get_static_path(self, root_dir, frame_index, index):
        img_path = self.get_image_path_layout(root_dir, frame_index, index)
        img_path = img_path.replace("image_02/data", 'road_256/road_256')
        return img_path

    def get_dynamic_path(self, root_dir, frame_index, ind):
        index = int(frame_index.split('road_labels/')[1].split('.')[0])
        img_path = os.path.join(root_dir, frame_index.replace(
            "road_labels", "vehicle_labels"))
        img_path = img_path.split('/00')[0]
        img_path = os.path.join(img_path, "%010d.png" % int(index+ind))
        return img_path

    def get_osm_path(self, root_dir):
        osm_file = np.random.choice(os.listdir(root_dir))
        osm_path = os.path.join(root_dir, osm_file)

        return osm_path

    def get_static_gt_path(self, root_dir, frame_index):
        path = os.path.join(
            root_dir,
            frame_index).replace(
            "road_bev",
            "road_gt")
        return path
    def get_static_K(self, root_dir, frame_index, ind):
        date = frame_index.split('/')[0]
        drive = frame_index.split('_sync')[0].split('_')[-1]
        basedir = '/CV/datasets/kitti_data/'
        data = pykitti.raw(basedir, date, drive)
        K = data.calib.K_cam2
        Tr_cam2_velo = data.calib.T_cam2_velo
        return K, Tr_cam2_velo
    # def get_static_Tr_cam2_velo(self, root_dir, frame_index, ind):
    #     date = frame_index.split('/')[0]
    #     drive = frame_index.split('_sync')[0].split('_')[-1]
    #     basedir = '/CV/datasets/kitti_data/'
    #     data = pykitti.raw(basedir, date, drive)
    #     # K = data.calib.K_cam2
    #     Tr_cam2_velo = data.calib.T_cam2_velo
    #     return Tr_cam2_velo
    def get_dynamic_gt_path(self, root_dir, frame_index):
        return self.get_dynamic_path(root_dir, frame_index)
    def creat_both_gt(self, path_static, path_dynamic):
        tvs = self.loader(path_static)
        try:
            tvd = self.loader(path_dynamic).convert('L').convert('RGB').resize((256, 256), pil.NEAREST)
            img_array = np.array(tvd)
            # print("))))))))))))))))))",img_array.shape)
            height,width,c = img_array.shape

            dst = np.zeros((height, width, 3))
            for h in range(0, height):
                for w in range(0, width):
                    (b, g, r) = img_array[h, w]
                    if (b, g, r) > (125, 125, 125):  # 白色
                        img_array[h, w] = (0, 0, 255)  # 蓝色
            img2 = pil.fromarray(np.uint8(img_array)).convert('RGB')
            # print("img2 size-----------------",img2.size)
            L, H = img2.size
            for h in range(H):
                for l in range(L):
                    dot = (l, h)
                    color_1 = img2.getpixel(dot)
                    if color_1 == (0, 0, 255):
                        tvs.putpixel(dot, color_1)
        except FileNotFoundError:
            tvs = tvs
        both_path = path_dynamic.replace("vehicle_labels", "both_gt")
        both_path_0 = both_path.rsplit("/",1)[0]
        if not os.path.exists(both_path_0):
            os.makedirs(both_path_0)
        if os.path.exists(both_path) and os.path.isdir(both_path):
            shutil.rmtree(both_path)

        # if not os.path.exists(both_path):
        tvs.save(both_path)
        # print("saving gt__________ ",both_path)
        # print(both_path)
        # return tvs

class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)
    def get_color_layout(self, root_dir, frame_index, index, do_flip):
        color = self.loader(self.get_image_path(root_dir, frame_index, index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    def get_image_path(self, root_dir, frame_index, ind):
        index = int(frame_index.split('road_dense128/')[1].split('.')[0])
        file_name = frame_index.replace("road_dense128", "image_2")
        img_path = os.path.join(root_dir, file_name)
        img_path = os.path.join(img_path.split('image_2')[0],"image_2")
        img_path = os.path.join(img_path, "%06d.png" % int(index + ind))
        return img_path
    def get_static(self, path, do_flip):
        tv = self.loader(path)
        if do_flip:
            tv = tv.transpose(pil.FLIP_LEFT_RIGHT)
        return tv.convert('L')
    def get_static_path(self, root_dir, frame_index, ind):
        index = int(frame_index.split('road_dense128/')[1].split('.')[0])
        img_path = os.path.join(root_dir, frame_index)
        sequence = img_path.split('/road_dense128')[0].split('sequences/')[1]
        img_path = os.path.join(img_path.split('road_dense128')[0],"road_dense128")
        img_path = os.path.join(img_path, "%06d.png" % int(index+ind))

        img_index = int(index+ind)
        basedir = '/CV/datasets/kitti_data/odometry/dataset'
        data = pykitti.odometry(basedir, sequence)
        K = data.calib.K_cam2
        Tr_cam2_velo = data.calib.T_cam2_velo
        return img_path, K, Tr_cam2_velo
    def get_static_K(self, root_dir, frame_index, ind):
        index = int(frame_index.split('road_dense128/')[1].split('.')[0])
        img_path = os.path.join(root_dir, frame_index)
        sequence = img_path.split('/road_dense128')[0].split('sequences/')[1]
        img_path = os.path.join(img_path.split('road_dense128')[0],"road_dense128")
        img_path = os.path.join(img_path, "%06d.png" % int(index+ind))

        img_index = int(index+ind)
        basedir = '/CV/datasets/kitti_data/odometry/dataset'
        data = pykitti.odometry(basedir, sequence)
        K = data.calib.K_cam2
        Tr_cam2_velo = data.calib.T_cam2_velo
        return K, Tr_cam2_velo
    # def get_static_Tr_cam2_velo(self, root_dir, frame_index, ind):
    #     index = int(frame_index.split('road_dense128/')[1].split('.')[0])
    #     img_path = os.path.join(root_dir, frame_index)
    #     sequence = img_path.split('/road_dense128')[0].split('sequences/')[1]
    #     img_path = os.path.join(img_path.split('road_dense128')[0],"road_dense128")
    #     img_path = os.path.join(img_path, "%06d.png" % int(index+ind))
    #
    #     img_index = int(index+ind)
    #     basedir = '/CV/datasets/kitti_data/odometry/dataset'
    #     data = pykitti.odometry(basedir, sequence)
    #     # K = data.calib.K_cam2
    #     Tr_cam2_velo = data.calib.T_cam2_velo
    #     return Tr_cam2_velo
    def get_depth_odom(self, root_dir, frame_index, do_flip):
        odom_to_raw_dict = {"00": "2011_10_03/2011_10_03_drive_0027",
                            "01": "2011_10_03/2011_10_03_drive_0042",
                            "02": "2011_10_03/2011_10_03_drive_0034",
                            "03": "2011_09_26/2011_09_26_drive_0067",
                            "04": "2011_09_30/2011_09_30_drive_0016",
                            "05": "2011_09_30/2011_09_30_drive_0018",
                            "06": "2011_09_30/2011_09_30_drive_0020",
                            "07": "2011_09_30/2011_09_30_drive_0027",
                            "08": "2011_09_30/2011_09_30_drive_0028",
                            "09": "2011_09_30/2011_09_30_drive_0033",
                            "10": "2011_09_30/2011_09_30_drive_0034"}
        index = int(frame_index.split('road_dense128/')[1].split('.')[0])
        files = frame_index.split('/road_dense128/')[0]
        calib_path = odom_to_raw_dict[files].split('/')[0]
        root = root_dir.split('/odometry')[0]
        calib_path = os.path.join(root,calib_path)
        velo_filename = os.path.join(
            root_dir,files,
            "velodyne/{:06d}.bin".format(int(index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, 2)
        # depth_gt = skimage.transform.resize(
        #     depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')
        im = pil.fromarray(depth_gt)
        depth_gt = np.array(im.resize(self.full_res_shape[::-1], pil.NEAREST)).astype(np.double)
        # depth_gt = scipy.misc.imresize(depth_gt, self.full_res_shape[::-1], "nearest")

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return np.array(depth_gt.copy())

class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

class KITTIObject(MonoDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """

    def __init__(self, *args, **kwargs):
        super(KITTIObject, self).__init__(*args, **kwargs)
        self.root_dir = "./data/object"

    def get_image_path(self, folder, frame_index, i):
        # if self.is_train == True:
        image_dir = os.path.join(folder, "training", 'image_2')
        # else:
        #     image_dir = os.path.join(folder, "testing", 'image_2')
        # print("******************",frame_index)
        index = int(frame_index)

        img_path = os.path.join(image_dir, "%06d.png" % (index + i))


        return img_path
    def get_color_layout(self, root_dir, frame_index, index, do_flip):
        color = self.loader(self.get_image_path(root_dir, frame_index, index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    def get_dynamic_path(self, root_dir, frame_index, i):
        # if self.is_train == True:
        tv_dir = os.path.join(root_dir, "training", 'vehicle_256')
        # else:
        #     tv_dir = os.path.join(root_dir, "testing", 'vehicle_256')
        index = int(frame_index)
        tv_path = os.path.join(tv_dir, "%06d.png" % (index + i))
        return tv_path
    def get_dynamic(self, path, do_flip):
        tv = self.loader(path)
        if do_flip:
            tv = tv.transpose(pil.FLIP_LEFT_RIGHT)
        return tv.convert('L')
    def get_dynamic_gt_path(self, root_dir, frame_index):
        return self.get_dynamic_path(root_dir, frame_index)

    def get_static_gt_path(self, root_dir, frame_index):
        pass
    def map_object_to_raw(self, frame_index):
        mapping = []
        mapping_file = "/CV/datasets/kitti_data/object/train_mapping.txt"
        with open(mapping_file, 'r') as f:
            for line in f:
                mapping.append(line)
        # print(result)

    def get_dynamic_K(self, root_dir, frame_index, mapping):
        # print(mapping)
        date = mapping.split(' ')[0]
        # print("date:", date)
        drive = mapping.split(' ')[1]
        # print("sync:", drive)
        drive = drive.split('_sync')[0]
        drive = drive.split('_')[-1]
        basedir = '/CV/datasets/kitti_data/'
        data = pykitti.raw(basedir, date, drive)
        K = data.calib.K_cam2
        Tr_cam2_velo = data.calib.T_cam2_velo
        return K, Tr_cam2_velo
