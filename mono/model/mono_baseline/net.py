from __future__ import absolute_import, division, print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
from .dice_loss import IoULoss, TverskyLoss, SoftDiceLoss
from .focal_loss import FocalLoss
from .boundary_loss import BDLoss
from .layers import SSIM, Backproject, Project, disp_to_depth, SE3
from .depth_encoder import DepthEncoder
from .depth_decoder import DepthDecoder
from .pose_encoder import PoseEncoder
from .pose_decoder import PoseDecoder
from ..registry import MONO
from .layout_model import Encoder, Decoder
from .CycledViewProjection import CycledViewProjection
from .CrossViewTransformer import CrossViewTransformer
import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import cv2
# from argoverse.utils.se3 import SE3
import pykitti
import os
#from thop import clever_format
#from thop import profile
import torchvision
from torchvision import transforms
from torchgeometry.core.imgwarp import warp_perspective
from torchgeometry.core.transformations import transform_points
np.set_printoptions(threshold=np.inf)
@MONO.register_module
class Baseline(nn.Module):
    def __init__(self, options):
        super(Baseline, self).__init__()
        self.opt = options
        # print("option keys-------------", self.opt.keys())
        self.num_input_frames = len(self.opt.frame_ids)
        self.DepthEncoder = DepthEncoder(self.opt.depth_num_layers,
                                         self.opt.depth_pretrained_path)
        self.DepthDecoder = DepthDecoder(self.DepthEncoder.num_ch_enc)
        self.PoseEncoder = PoseEncoder(self.opt.pose_num_layers,
                                       self.opt.pose_pretrained_path,
                                       num_input_images=2)
        self.PoseDecoder = PoseDecoder(self.PoseEncoder.num_ch_enc)
        self.LayoutEncoder = Encoder(self.opt.depth_num_layers, True)
        self.CycledViewProjection = CycledViewProjection(in_dim=self.opt.occ_map_size//32)
        self.CrossViewTransformer = CrossViewTransformer(128)
        self.LayoutDecoder = Decoder(
            self.LayoutEncoder.resnet_encoder.num_ch_enc, self.opt.num_class)
        self.LayoutTransformDecoder = Decoder(
            self.LayoutEncoder.resnet_encoder.num_ch_enc, self.opt.num_class, "transform_decoder")

        # self.LayoutEncoderB = Encoder(self.opt.depth_num_layers, True)
        self.CycledViewProjectionB = CycledViewProjection(in_dim=self.opt.occ_map_size // 32)
        self.CrossViewTransformerB = CrossViewTransformer(128)
        self.LayoutDecoderB = Decoder(
            self.LayoutEncoder.resnet_encoder.num_ch_enc, self.opt.num_class)
        self.LayoutTransformDecoderB = Decoder(
            self.LayoutEncoder.resnet_encoder.num_ch_enc, self.opt.num_class, "transform_decoder")

        self.ssim = SSIM()
        self.backproject = Backproject(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)
        self.project_3d = Project(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)
        self.weight = {"static": self.opt.static_weight, "dynamic": self.opt.dynamic_weight}


    def forward(self, inputs):
        depth_feature = self.DepthEncoder(inputs["color_aug", 0, 0])

        outputs = self.DepthDecoder(depth_feature)

        outputs.update(self.predict_layout(inputs, depth_feature)[0])
        encoder_features = self.predict_layout(inputs, depth_feature)[1]
        outputs.update(self.predict_layoutB(inputs, depth_feature, encoder_features))
        # outputs.update(self.predict_poses(inputs))
        if self.training:
            outputs.update(self.predict_poses(inputs))
            loss_dict = self.compute_losses(inputs, outputs)
            return outputs, loss_dict

        return outputs

    def robust_l1(self, pred, target):
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

    def compute_reprojection_loss(self, pred, target):
        photometric_loss = self.robust_l1(pred, target).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        loss_dict = {}
        if self.opt["type"] == "static_raw" or self.opt["type"] == "static" or self.opt["type"] == "Argo_static" or self.opt["type"] == "Argo_both":

            weightS = torch.Tensor([1., self.weight["static"]])
        if self.opt["type"] == "dynamic" or self.opt["type"] == "Argo_dynamic" or self.opt["type"] == "Argo_both":
            weightD = torch.Tensor([1., self.weight["dynamic"]])
        if self.opt["type"] == "static" or self.opt["type"] == "static_raw" or self.opt["type"] == "Argo_static":
            scale_label = self.get_scale_label_static(inputs, self.opt)
        elif self.opt["type"] == "dynamic" or self.opt["type"] == "Argo_dynamic":
            scale_label = self.get_scale_label_dynamic(inputs, self.opt)
        elif self.opt["type"] == "Argo_both":
            scale_label = self.get_scale_label_both(inputs, self.opt)
        loss_dict["topview_loss"] = 0
        loss_dict["transform_topview_loss"] = 0
        loss_dict["transform_loss"] = 0
        loss_dict["topview_lossB"] = 0
        loss_dict["transform_topview_lossB"] = 0
        loss_dict["transform_lossB"] = 0
        loss_dict["topview_loss"] = self.compute_topview_loss(
            outputs["topview"],
            inputs["bothS",0,0],
            weightS, self.opt)
        loss_dict["transform_topview_loss"] = self.compute_topview_loss(
            outputs["transform_topview"],
            inputs["bothS",0,0],
            weightS, self.opt)
        loss_dict["transform_loss"] = self.compute_transform_losses(
            outputs["features"],
            outputs["retransform_features"])
        loss_dict["layout_loss"] = loss_dict["topview_loss"] + 0.001 * loss_dict["transform_loss"] \
                         + 1 * loss_dict["transform_topview_loss"]
        loss_dict["topview_lossB"] = self.compute_topview_lossB(
            outputs["topviewB"],
            inputs["bothD", 0, 0],
            weightD, self.opt)
        loss_dict["transform_topview_lossB"] = self.compute_topview_lossB(
            outputs["transform_topviewB"],
            inputs["bothD", 0, 0],
            weightD, self.opt)
        loss_dict["transform_lossB"] = self.compute_transform_losses(
            outputs["featuresB"],
            outputs["retransform_featuresB"])
        loss_dict["layout_lossB"] = loss_dict["topview_lossB"] + 0.001 * loss_dict["transform_lossB"] \
                                   + 1 * loss_dict["transform_topview_lossB"]
        for scale in self.opt.scales:
            """
            initialization
            """
   #         start3 = time.time()

            disp = outputs[("disp", 0, scale)]
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth
            target = inputs[("color", 0, 0)]
            reprojection_losses = []

            """
            reconstruction
            """
            outputs = self.generate_images_pred(inputs, outputs, scale)

            """
            automask
            """
            if self.opt.automask:
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, 0)]
                    identity_reprojection_loss = self.compute_reprojection_loss(pred, target)
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 1e-5
                    reprojection_losses.append(identity_reprojection_loss)

            """
            minimum reconstruction loss
            """
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_loss = torch.cat(reprojection_losses, 1)

            min_reconstruct_loss, outputs[("min_index", scale)] = torch.min(reprojection_loss, dim=1)
            loss_dict[('min_reconstruct_loss', scale)] = min_reconstruct_loss.mean()/len(self.opt.scales)
            scale_loss = self.get_scale_loss(inputs, outputs, scale, scale_label)
            loss_dict[('scale_loss', scale)] = self.opt.scale_weight * scale_loss / (2 ** scale) / len(
                self.opt.scales)
            """
            disp mean normalization
            """
            if self.opt.disp_norm:
                mean_disp = disp.mean(2, True).mean(3, True)
                disp = disp / (mean_disp + 1e-7)

            """
            smooth loss
            """
            smooth_loss = self.get_smooth_loss(disp, target)
            loss_dict[('smooth_loss', scale)] = self.opt.smoothness_weight * smooth_loss / (2 ** scale)/len(self.opt.scales)

        return loss_dict
    def get_scale_loss(self, inputs, outputs, scale, scale_label):
        depth_pred = outputs[("depth", 0, scale)]
        shape = scale_label.shape[2:4]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, shape, mode="bilinear", align_corners=False), 1e-3, 80)

        depth_gt = scale_label
        mask = depth_gt > 0
        if self.opt["type"] == "static_raw":# or self.opt["type"] == "static":
        # garg/eigen crop
            crop_mask = torch.zeros_like(mask)
            crop_mask[:, :, 153:371, 44:1197] = 1
            mask = mask * crop_mask

        depth_gt = torch.masked_select(depth_gt, mask)
        depth_pred = torch.masked_select(depth_pred, mask)
        # depth_pred = depth_pred[mask]
        abs_rel_loss = torch.mean(torch.abs(depth_gt - depth_pred) / depth_gt)
        return abs_rel_loss
    def get_scale_label_static(self, inputs, opt):
        # if there is  only road label, then the intersection of the assumption region and road region is used as scale label
        # the assumption region is a rectangle area in front of the ego car
        img_front = inputs[("color", 0, -1)]
        mapsize = opt.occ_map_size
        # img_front shape [128, 128, 3]
        # img batch [8, 3, 128, 128]
        height, width = img_front.shape[2:4]

        fig = plt.figure(figsize=(15, 10))
        bev_img_layout = inputs[("bothS", 0, 0)]
        resolution = 40 / mapsize
        resolution1 = mapsize / 40
        if bev_img_layout.shape[2]!= mapsize or bev_img_layout.shape[3]!= mapsize:
            raise ValueError("The shape of both label is not ", mapsize)
        batchsize = bev_img_layout.shape[0]
        h = w = mapsize
        if opt.split == "argo":
            z = torch.arange(mapsize, 0, step=-1).view(1, 1, h, 1).repeat(batchsize, 1, 1, w) * (40 / mapsize) - 1.9
        elif opt.split == "raw" or "odometry":

            z = torch.arange(mapsize, 0, step=-1).view(1, 1, h, 1).repeat(batchsize, 1, 1, w) * (40 / mapsize) - 0.27
        bev_layout_distance_label = z.cuda()
        points_real = [(round(18 * resolution1), round(31 * resolution1)),
                       (round(22 * resolution1), round(31 * resolution1)),
                       (round(18 * resolution1), round(33 * resolution1)),
                       (round(22 * resolution1), round(33 * resolution1))]
        bev_img_layout = torch.fliplr(bev_img_layout)
        bev_layout_distance_label = torch.fliplr(bev_layout_distance_label)
        bev_img_layout = torchvision.transforms.functional.rotate(bev_img_layout, angle=270)
        bev_layout_distance_label = torchvision.transforms.functional.rotate(bev_layout_distance_label, angle=270)
        points_real_rot = [[mapsize - points_real[3][1] - 1, points_real[0][0] - 1],
                           [mapsize - points_real[3][1] + (points_real[2][1] - points_real[1][1]) - 1,
                            points_real[0][0] - 1],
                           [mapsize - points_real[3][1] - 1, points_real[1][0] - 1],
                           [mapsize - points_real[3][1] + (points_real[2][1] - points_real[1][1]) - 1,
                            points_real[1][0] - 1]]

        K = inputs[("odometry_K", 0, 0)][:,:3,:3]
        batchsize = K.shape[0]
        Tr_cam2_velo = inputs[("Tr_cam2_velo", 0, 0)]
        camera_SE3_egovehicleO = Tr_cam2_velo
        camera_R_egovehicle = camera_SE3_egovehicleO[:, :3, :3]
        camera_t_egovehicle = camera_SE3_egovehicleO[:, :3, 3]
        camera_SE3_egovehicle = SE3(rotation=camera_R_egovehicle, translation=camera_t_egovehicle)
        if opt.split == "argo":
            HEIGHT_FROM_GROUND = 0.33
        elif opt.split == "raw" or opt.split == "odometry":
            HEIGHT_FROM_GROUND = 1.73  # in meters,
        ground_rotation = torch.eye(3).repeat(batchsize,1,1)
        ground_translation = torch.Tensor([0, 0, HEIGHT_FROM_GROUND]).repeat(batchsize,1)
        ground_SE3_egovehicle = SE3(rotation=ground_rotation, translation=ground_translation)
        egovehicle_SE3_ground = ground_SE3_egovehicle(None, "inverse")

        camera_SE3_ground = camera_SE3_egovehicle(egovehicle_SE3_ground,"right_multiply_with_se3")
        img_H_ground=self.homography_from_calibration(camera_SE3_ground, K)
        ground_H_img = torch.linalg.inv(img_H_ground)
        LATERAL_EXTENT = 20  # look 20 meters left and right
        FORWARD_EXTENT = 40  # look 40 meters ahead

        # in meters/px
        out_width = int(FORWARD_EXTENT / resolution)
        out_height = int(LATERAL_EXTENT * 2 / resolution)

        RESCALING = 1 / resolution  # pixels/meter, if rescaling=1, then 1 px/1 meter
        SHIFT = int(out_width // 2)
        shiftedground_H_ground = torch.Tensor(
            [
                [RESCALING, 0, 0],
                [0, RESCALING, SHIFT],
                [0, 0, 1]
            ]).repeat(batchsize, 1, 1).cuda()
        shiftedground_H_img = torch.bmm(shiftedground_H_ground, ground_H_img)
        restore_front_layout = warp_perspective(bev_img_layout, torch.linalg.inv(shiftedground_H_img),
                                                   dsize=(height, width))
        restore_front_bev_layout_distance_label = warp_perspective(bev_layout_distance_label,
                                                                      torch.linalg.inv(shiftedground_H_img),
                                                                      dsize=(height, width))

        layout_mask = restore_front_layout
        points_real_rot = np.asarray(points_real_rot)
        points_real_rot = torch.tensor(points_real_rot,dtype=torch.float32).repeat(batchsize,1,1).cuda()
        newPointsO = torch.round(transform_points(torch.linalg.inv(shiftedground_H_img), points_real_rot)).int()#(B,4,3)
        newPoints = newPointsO.cpu()
        pts = np.array(
            [[newPoints[0][0][0], newPoints[0][0][1]], [newPoints[0][2][0], newPoints[0][2][1]], [newPoints[0][3][0], newPoints[0][3][1]],
             [newPoints[0][1][0], newPoints[0][1][1]]])
        pts = np.round(pts).astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        imgshape = [img_front.shape[2:4][0], img_front.shape[2:4][1], 3]
        img_zero = np.zeros(imgshape, dtype=np.uint8)
        restore_front_layout = cv2.fillConvexPoly(img_zero, pts, (0, 255, 255), 1)
        img_gray_roi_mask_triangle = cv2.cvtColor(restore_front_layout, cv2.COLOR_RGB2GRAY)
        img_gray_roi_mask_triangle[img_gray_roi_mask_triangle > 0] = 255
        img_gray_roi_mask_triangle = torch.tensor(img_gray_roi_mask_triangle).repeat(batchsize,1,1).unsqueeze(1).cuda()
        layout_mask = layout_mask.type_as(img_gray_roi_mask_triangle)
        A_and_B = torch.bitwise_and(layout_mask, img_gray_roi_mask_triangle)
        restore_front_bev_layout_distance_label = restore_front_bev_layout_distance_label * A_and_B
        return restore_front_bev_layout_distance_label
    def get_scale_label_dynamic(self, inputs, opt):
        # if there is  only object label, then the assumption region is used as scale label
        # the assumption region is a rectangle area in front of the ego car

        img_front = inputs[("color", 0, -1)]
        # img_front shape [128, 128, 3]
        # img batch [8, 3, 128, 128]
        mapsize = opt.occ_map_size
        height, width = img_front.shape[2:4]
        bev_img_layout = inputs[("bothS", 0, 0)]
        resolution = 40 / mapsize
        resolution1 = mapsize / 40
        if bev_img_layout.shape[2]!= mapsize or bev_img_layout.shape[3]!= mapsize:
            raise ValueError("The shape of both label is not ", mapsize)
        batchsize = bev_img_layout.shape[0]
        h = w = mapsize
        if opt.split == "argo":
            z = torch.arange(mapsize, 0, step=-1).view(1, 1, h, 1).repeat(batchsize, 1, 1, w) * (40 / mapsize) - 1.9
        elif opt.split == "raw" or "odometry":
            z = torch.arange(mapsize, 0, step=-1).view(1, 1, h, 1).repeat(batchsize, 1, 1, w) * (40 / mapsize)  # - 0.27
        bev_layout_distance_label = z.cuda()
        points_real = [(round(18 * resolution1), round(31 * resolution1)),
                  (round(22 * resolution1), round(31 * resolution1)),
                  (round(18 * resolution1), round(33 * resolution1)),
                  (round(22 * resolution1), round(33 * resolution1))]
        bev_layout_distance_label = torch.fliplr(bev_layout_distance_label)
        bev_layout_distance_label = torchvision.transforms.functional.rotate(bev_layout_distance_label, angle=270)
        points_real_rot = [[mapsize - points_real[3][1] - 1, points_real[0][0] - 1],
                           [mapsize - points_real[3][1] + (points_real[2][1] - points_real[1][1]) - 1,
                            points_real[0][0] - 1],
                           [mapsize - points_real[3][1] - 1, points_real[1][0] - 1],
                           [mapsize - points_real[3][1] + (points_real[2][1] - points_real[1][1]) - 1,
                            points_real[1][0] - 1]]
        K = inputs[("odometry_K", 0, 0)][:, :3, :3]
        batchsize = K.shape[0]
        Tr_cam2_velo = inputs[("Tr_cam2_velo", 0, 0)]
        camera_SE3_egovehicleO = Tr_cam2_velo
        camera_R_egovehicle = camera_SE3_egovehicleO[:, :3, :3]
        camera_t_egovehicle = camera_SE3_egovehicleO[:, :3, 3]
        camera_SE3_egovehicle = SE3(rotation=camera_R_egovehicle, translation=camera_t_egovehicle)

        if opt.split == "argo":
            HEIGHT_FROM_GROUND = 0.33
        elif opt.split == "raw" or "odometry":
            HEIGHT_FROM_GROUND = 1.73  # in meters,
        ground_rotation = torch.eye(3).repeat(batchsize,1,1)

        ground_translation = torch.Tensor([0, 0, HEIGHT_FROM_GROUND]).repeat(batchsize,1)

        ground_SE3_egovehicle = SE3(rotation=ground_rotation, translation=ground_translation)
        egovehicle_SE3_ground = ground_SE3_egovehicle(None, "inverse")
        camera_SE3_ground = camera_SE3_egovehicle(egovehicle_SE3_ground,"right_multiply_with_se3")
        img_H_ground=self.homography_from_calibration(camera_SE3_ground, K)
        ground_H_img = torch.linalg.inv(img_H_ground)
        LATERAL_EXTENT = 20  # look 20 meters left and right
        FORWARD_EXTENT = 40  # look 40 meters ahead

        # in meters/px
        out_width = int(FORWARD_EXTENT / resolution)
        out_height = int(LATERAL_EXTENT * 2 / resolution)

        RESCALING = 1 / resolution  # pixels/meter, if rescaling=1, then 1 px/1 meter
        SHIFT = int(out_width // 2)
        shiftedground_H_ground = torch.Tensor(
            [
                [RESCALING, 0, 0],
                [0, RESCALING, SHIFT],
                [0, 0, 1]
            ]).repeat(batchsize, 1, 1).cuda()

        shiftedground_H_img = torch.bmm(shiftedground_H_ground, ground_H_img)
        restore_front_bev_layout_distance_label = warp_perspective(bev_layout_distance_label,
                                                                      torch.linalg.inv(shiftedground_H_img),
                                                                      dsize=(height, width))

        points_real_rot = np.asarray(points_real_rot)
        points_real_rot = torch.tensor(points_real_rot,dtype=torch.float32).repeat(batchsize,1,1).cuda()
        newPointsO = torch.round(transform_points(torch.linalg.inv(shiftedground_H_img), points_real_rot)).int()#(B,4,3)
        newPoints = newPointsO.cpu()
        pts = np.array(
            [[newPoints[0][0][0], newPoints[0][0][1]], [newPoints[0][2][0], newPoints[0][2][1]], [newPoints[0][3][0], newPoints[0][3][1]],
             [newPoints[0][1][0], newPoints[0][1][1]]])
        pts = np.round(pts).astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        imgshape = [img_front.shape[2:4][0], img_front.shape[2:4][1], 3]
        img_zero = np.zeros(imgshape, dtype=np.uint8)
        restore_front_layout = cv2.fillConvexPoly(img_zero, pts, (0, 255, 255), 1)
        img_gray_roi_mask_triangle = cv2.cvtColor(restore_front_layout, cv2.COLOR_RGB2GRAY)
        img_gray_roi_mask_triangle[img_gray_roi_mask_triangle > 0] = 1
        img_gray_roi_mask_triangle = torch.tensor(img_gray_roi_mask_triangle).repeat(batchsize,1,1).unsqueeze(1).cuda()
        restore_front_bev_layout_distance_label = restore_front_bev_layout_distance_label * img_gray_roi_mask_triangle
        return restore_front_bev_layout_distance_label
    def get_scale_label_both(self, inputs, opt):
        # if there are both labels, then the object region is subtracted from the road region, which is pre-computed as "both_dynamic"
        # and the remained region is used for the scale label
        # for KITTI dataset, the both label can also be computed by using pretrained model on other datasets
        # (e.g. models trained on KITTI Object to help generate the object region)
        img_front = inputs[("color", 0, -1)]
        mapsize = opt.occ_map_size
        # img_front shape [128, 128, 3]
        # img batch [8, 3, 128, 128]
        height, width = img_front.shape[2:4]

        fig = plt.figure(figsize=(15, 10))
        bev_img_layout = inputs[("both_dynamic", 0, 0)]
        resolution = 40 / mapsize
        resolution1 = mapsize / 40
        if bev_img_layout.shape[2] != mapsize or bev_img_layout.shape[3] != mapsize:
            raise ValueError("The shape of both label is not ", mapsize)
        batchsize = bev_img_layout.shape[0]
        h = w = mapsize
        if opt.split == "argo":
            z = torch.arange(mapsize, 0, step=-1).view(1, 1, h, 1).repeat(batchsize, 1, 1, w) * (40 / mapsize) - 1.9
        elif opt.split == "raw" or "odometry":

            z = torch.arange(mapsize, 0, step=-1).view(1, 1, h, 1).repeat(batchsize, 1, 1, w) * (40 / mapsize) - 0.27
        bev_layout_distance_label = z.cuda()
        bev_img_layout = torch.fliplr(bev_img_layout)
        bev_layout_distance_label = torch.fliplr(bev_layout_distance_label)
        bev_img_layout = torchvision.transforms.functional.rotate(bev_img_layout, angle=270)
        bev_layout_distance_label = torchvision.transforms.functional.rotate(bev_layout_distance_label, angle=270)

        K = inputs[("odometry_K", 0, 0)][:, :3, :3]
        batchsize = K.shape[0]
        Tr_cam2_velo = inputs[("Tr_cam2_velo", 0, 0)]
        camera_SE3_egovehicleO = Tr_cam2_velo
        camera_R_egovehicle = camera_SE3_egovehicleO[:, :3, :3]
        camera_t_egovehicle = camera_SE3_egovehicleO[:, :3, 3]
        camera_SE3_egovehicle = SE3(rotation=camera_R_egovehicle, translation=camera_t_egovehicle)
        if opt.split == "argo":
            HEIGHT_FROM_GROUND = 0.33
        elif opt.split == "raw" or opt.split == "odometry":
            HEIGHT_FROM_GROUND = 1.73  # in meters,
        ground_rotation = torch.eye(3).repeat(batchsize, 1, 1)
        ground_translation = torch.Tensor([0, 0, HEIGHT_FROM_GROUND]).repeat(batchsize, 1)
        ground_SE3_egovehicle = SE3(rotation=ground_rotation, translation=ground_translation)
        egovehicle_SE3_ground = ground_SE3_egovehicle(None, "inverse")

        camera_SE3_ground = camera_SE3_egovehicle(egovehicle_SE3_ground, "right_multiply_with_se3")
        img_H_ground = self.homography_from_calibration(camera_SE3_ground, K)
        ground_H_img = torch.linalg.inv(img_H_ground)
        LATERAL_EXTENT = 20  # look 20 meters left and right
        FORWARD_EXTENT = 40  # look 40 meters ahead

        # in meters/px
        out_width = int(FORWARD_EXTENT / resolution)
        out_height = int(LATERAL_EXTENT * 2 / resolution)

        RESCALING = 1 / resolution  # pixels/meter, if rescaling=1, then 1 px/1 meter
        SHIFT = int(out_width // 2)
        shiftedground_H_ground = torch.Tensor(
            [
                [RESCALING, 0, 0],
                [0, RESCALING, SHIFT],
                [0, 0, 1]
            ]).repeat(batchsize, 1, 1).cuda()
        shiftedground_H_img = torch.bmm(shiftedground_H_ground, ground_H_img)
        restore_front_layout = warp_perspective(bev_img_layout, torch.linalg.inv(shiftedground_H_img),
                                                dsize=(height, width))
        restore_front_bev_layout_distance_label = warp_perspective(bev_layout_distance_label,
                                                                   torch.linalg.inv(shiftedground_H_img),
                                                                   dsize=(height, width))

        layout_mask = restore_front_layout
        restore_front_bev_layout_distance_label = restore_front_bev_layout_distance_label * layout_mask
        return restore_front_bev_layout_distance_label
    def SE3(self,rotation,translation):
        assert rotation[0].shape == (3, 3)
        assert translation[0].shape == (3,)
        # self.rotation = rotation
        # self.translation = translation
        batch = rotation.shape[0]
        transform_matrix = torch.eye(4).repeat(batch, 1, 1).cuda()
        # print("self.transform_matrix", self.transform_matrix)
        transform_matrix[:, :3, :3] = rotation
        transform_matrix[:, :3, 3] = translation
        se3_dict = {"rotation":rotation, "translation":translation, "transform_matrix":transform_matrix}
        return se3_dict
    def inverse(self, se3_dict):

        """Return the inverse of the current SE3 transformation.

        For example, if the current object represents target_SE3_src, we will return instead src_SE3_target.

        Returns:
            src_SE3_target: instance of SE3 class, representing
                inverse of SE3 transformation target_SE3_src
        """
        # rotation [3,3]
        # tensor [8, 3. 3]
        rotation = se3_dict["rotation"].transpose(1, 2)
        # print("rotation-----",rotation.shape)
        # print("-self.translation-----", (-se3_dict["translation"]).shape)
        _translation = -se3_dict["translation"].unsqueeze(2)
        translation = torch.bmm(rotation, _translation)
        # print("translation-----", translation.shape)
        return self.SE3(rotation=rotation, translation=translation.squeeze(2))
    def right_multiply_with_se3(self,se3_dict, right_se3):
        return self.compose(se3_dict, right_se3)

    def compose(self, se3_dict, right_se3):
        chained_transform_matrix = torch.bmm(se3_dict["transform_matrix"], right_se3["transform_matrix"])
        chained_se3 = self.SE3(
            rotation=chained_transform_matrix[:, :3, :3],
            translation=chained_transform_matrix[:, :3, 3],
        )
        return chained_se3
    def rotate_points(self,points, Matrix):
        warpPoints = []
        for i in range(len(points)):
            point = points[i]
            point = torch.from_numpy(np.array(point).reshape(1, -1, 2).astype(np.float32)).cuda()  # 二维变三维， 整形转float型， 一个都不能少
            #cv2.perspectiveTransform
            new_points = transform_points(Matrix, point)
            warpPoints.append(new_points)#.round().astype(np.int32).squeeze(0).squeeze(0))
        # print("warpPoints----:", warpPoints)
        return warpPoints

    def homography_from_calibration(self,camera_SE3_ground: SE3, K: torch.Tensor) -> torch.Tensor:
        """
        See Hartley Zisserman, Section 8.1.1
        """
        # print("camera_SE3_ground device",camera_SE3_ground.transform_matrix.device)
        batch = K.shape[0]
        # print("r1**************", camera_SE3_ground.transform_matrix[:, :3, 0].shape)
        r1 = camera_SE3_ground.transform_matrix[:, :3, 0].reshape(batch, -1, 1)
        # print("r1**************", r1.shape)
        r2 = camera_SE3_ground.transform_matrix[:, :3, 1].reshape(batch, -1, 1)
        t = camera_SE3_ground.transform_matrix[:, :3, 3].reshape(batch, -1, 1)
        # print("k in here",K.shape,K.device)
        # print("torch.cat([r1, r2, t])",torch.cat([r1, r2, t], dim=2).shape,torch.cat([r1, r2, t], dim=2).device)
        img_H_ground = torch.bmm(K, torch.cat([r1, r2, t], dim=2))
        return img_H_ground
    # def compute_topview_loss(self, outputs, true_top_view, weight):
    #     generated_top_view = outputs
    #     # print("generated_top_view", generated_top_view.shape)
    #     true_top_view = torch.squeeze(true_top_view.long())
    #     # print("true_top_view",true_top_view.shape)
    #     if true_top_view.shape[0] == 256:
    #         true_top_view = torch.unsqueeze(true_top_view,dim=0)
    #     loss = nn.CrossEntropyLoss(weight=weight.cuda())
    #     output = loss(generated_top_view, true_top_view)
    #     return output.mean()
    def compute_topview_loss(self, outputs, true_top_view, weight, opt):
        generated_top_view = outputs
        #print("generated_top_view", generated_top_view.shape)
        true_top_view = torch.squeeze(true_top_view.long())
        if true_top_view.shape[0] == 256:
            true_top_view = torch.unsqueeze(true_top_view,dim=0)
        #print("true_top_view",true_top_view.shape)
        loss1 = nn.CrossEntropyLoss(weight=weight.cuda())
        if opt.loss_type == 'iou':
            softmax_helper = lambda x: F.softmax(x, 1)
            loss = IoULoss(apply_nonlin=softmax_helper)
        elif opt.loss_type == 'dice':
            softmax_helper = lambda x: F.softmax(x, 1)
            loss = SoftDiceLoss(apply_nonlin=softmax_helper)
        elif opt.loss_type == 'focal':
            softmax_helper = lambda x: F.softmax(x, 1)
            loss = FocalLoss(apply_nonlin=softmax_helper)
        elif opt.loss_type == 'tversky':
            softmax_helper = lambda x: F.softmax(x, 1)
            loss = TverskyLoss(apply_nonlin=softmax_helper)
        if opt.loss2_type == 'boundary':
            loss2 = BDLoss()
        if opt.loss_sum == 1:
            output = loss(generated_top_view, true_top_view) * opt.loss_weightS
        elif opt.loss_sum == 2:
            output = loss(generated_top_view, true_top_view)*opt.loss_weightS + \
                     loss2(generated_top_view, true_top_view)*opt.loss2_weightS
        elif opt.loss_sum == 3:
            output = loss(generated_top_view, true_top_view) * opt.loss_weightS + \
                     loss1(generated_top_view, true_top_view) +\
                     loss2(generated_top_view, true_top_view) * opt.loss2_weightS
        return output.mean()
    def compute_topview_lossB(self, outputs, true_top_view, weight, opt):
        generated_top_view = outputs
        #print("generated_top_view", generated_top_view.shape)
        true_top_view = torch.squeeze(true_top_view.long())
        if true_top_view.shape[0] == 256:
            true_top_view = torch.unsqueeze(true_top_view,dim=0)
        #print("true_top_view",true_top_view.shape)
        loss1 = nn.CrossEntropyLoss(weight=weight.cuda())
        if opt.loss_type == 'iou':
            softmax_helper = lambda x: F.softmax(x, 1)
            loss = IoULoss(apply_nonlin=softmax_helper)
        elif opt.loss_type == 'dice':
            softmax_helper = lambda x: F.softmax(x, 1)
            loss = SoftDiceLoss(apply_nonlin=softmax_helper)
        elif opt.loss_type == 'focal':
            softmax_helper = lambda x: F.softmax(x, 1)
            loss = FocalLoss(apply_nonlin=softmax_helper)
        elif opt.loss_type == 'tversky':
            softmax_helper = lambda x: F.softmax(x, 1)
            loss = TverskyLoss(apply_nonlin=softmax_helper)
        if opt.loss2_type == 'boundary':
            loss2 = BDLoss()
        if opt.loss_sum == 1:
            output = loss(generated_top_view, true_top_view) * opt.loss_weight
        elif opt.loss_sum == 2:
            output = loss(generated_top_view, true_top_view)*opt.loss_weight + \
                     loss2(generated_top_view, true_top_view)*opt.loss2_weight
        elif opt.loss_sum == 3:
            output = loss(generated_top_view, true_top_view) * opt.loss_weight + \
                     loss1(generated_top_view, true_top_view) +\
                     loss2(generated_top_view, true_top_view) * opt.loss2_weight
        return output.mean()

    def compute_transform_losses(self, outputs, retransform_output):
        self.L1Loss = nn.L1Loss()
        loss = self.L1Loss(outputs, retransform_output)
        return loss
    def disp_to_depth(self, disp, min_depth, max_depth):
        min_disp = 1 / max_depth  # 0.01
        max_disp = 1 / min_depth  # 10
        scaled_disp = min_disp + (max_disp - min_disp) * disp  # (10-0.01)*disp+0.01
        depth = 1 / scaled_disp
        return scaled_disp, depth

    def predict_poses(self, inputs):
        outputs = {}
        pose_feats = {f_i: F.interpolate(inputs["color_aug", f_i, 0], [192, 640], mode="bilinear", align_corners=False) for f_i in self.opt.frame_ids}
        for f_i in self.opt.frame_ids[1:]:
            if not f_i == "s":
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]
                pose_inputs = self.PoseEncoder(torch.cat(pose_inputs, 1))
                axisangle, translation = self.PoseDecoder(pose_inputs)
                outputs[("cam_T_cam", 0, f_i)] = self.transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        return outputs

    def predict_layout(self, inputs, depth_feature):
        outputs = {}
        features = self.LayoutEncoder(inputs["color_aug", 0, 0])
        encoder_features = features
        # Cross-view Transformation Module
        outputs["origin_features"] = features
        transform_feature, retransform_features = self.CycledViewProjection(features)
        # print("transform_feature#####################",transform_feature.shape)
        # print("retransform_features#####################", retransform_features.shape)
        depth_feature = depth_feature[-1]
        # print("depth_feature#####################", depth_feature.shape)
        features, energy, atten = self.CrossViewTransformer(features, transform_feature, retransform_features,depth_feature)

        outputs["topview"] = self.LayoutDecoder(features)
        outputs["transform_topview"] = self.LayoutTransformDecoder(transform_feature)
        outputs["features"] = features
        outputs["features_road"] = outputs["features"]
        outputs["transform_feature_road"] = transform_feature
        outputs["retransform_features"] = retransform_features
        outputs["retransform_features_road"] = outputs["retransform_features"]
        outputs["cv_attn_road"] = energy
        outputs["cm_attn_road"] = atten
        return outputs, encoder_features
    def predict_layoutB(self, inputs, depth_feature, encoder_features):
        outputs = {}
        features = encoder_features

        # Cross-view Transformation Module

        transform_feature, retransform_features = self.CycledViewProjectionB(features)
        # print("transform_feature#####################",transform_feature.shape)
        # print("retransform_features#####################", retransform_features.shape)
        depth_feature = depth_feature[-1]
        # print("depth_feature#####################", depth_feature.shape)
        features, energy, atten  = self.CrossViewTransformerB(features, transform_feature, retransform_features,depth_feature)

        outputs["topviewB"] = self.LayoutDecoderB(features)
        outputs["transform_topviewB"] = self.LayoutTransformDecoderB(transform_feature)
        outputs["featuresB"] = features
        outputs["transform_feature_car"] = transform_feature
        outputs["features_car"] = outputs["featuresB"]
        outputs["retransform_featuresB"] = retransform_features
        outputs["retransform_features_car"] = outputs["retransform_featuresB"]
        outputs["cv_attn_car"] = energy
        outputs["cm_attn_car"] = atten
        return outputs
    def generate_images_pred(self, inputs, outputs, scale):
        disp = outputs[("disp", 0, scale)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        _, depth = self.disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            if frame_id == "s":
                T = inputs["stereo_T"]
            else:
                T = outputs[("cam_T_cam", 0, frame_id)]
            cam_points = self.backproject(depth, inputs[("inv_K",0)])
            pix_coords = self.project_3d(cam_points, inputs[("K",0)], T)#[b,h,w,2]
            outputs[("color", frame_id, scale)] = F.grid_sample(inputs[("color", frame_id, 0)], pix_coords, padding_mode="border")
        return outputs

    def transformation_from_parameters(self, axisangle, translation, invert=False):
        R = self.rot_from_axisangle(axisangle)
        t = translation.clone()
        if invert:
            R = R.transpose(1, 2)
            t *= -1
        T = self.get_translation_matrix(t)
        if invert:
            M = torch.matmul(R, T)
        else:
            M = torch.matmul(T, R)
        return M

    def get_translation_matrix(self, translation_vector):
        T = torch.zeros(translation_vector.shape[0], 4, 4).cuda()
        t = translation_vector.contiguous().view(-1, 3, 1)
        T[:, 0, 0] = 1
        T[:, 1, 1] = 1
        T[:, 2, 2] = 1
        T[:, 3, 3] = 1
        T[:, :3, 3, None] = t
        return T

    def rot_from_axisangle(self, vec):
        angle = torch.norm(vec, 2, 2, True)
        axis = vec / (angle + 1e-7)
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        C = 1 - ca
        x = axis[..., 0].unsqueeze(1)
        y = axis[..., 1].unsqueeze(1)
        z = axis[..., 2].unsqueeze(1)
        xs = x * sa
        ys = y * sa
        zs = z * sa
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC
        rot = torch.zeros((vec.shape[0], 4, 4)).cuda()
        rot[:, 0, 0] = torch.squeeze(x * xC + ca)
        rot[:, 0, 1] = torch.squeeze(xyC - zs)
        rot[:, 0, 2] = torch.squeeze(zxC + ys)
        rot[:, 1, 0] = torch.squeeze(xyC + zs)
        rot[:, 1, 1] = torch.squeeze(y * yC + ca)
        rot[:, 1, 2] = torch.squeeze(yzC - xs)
        rot[:, 2, 0] = torch.squeeze(zxC - ys)
        rot[:, 2, 1] = torch.squeeze(yzC + xs)
        rot[:, 2, 2] = torch.squeeze(z * zC + ca)
        rot[:, 3, 3] = 1
        return rot

    def get_smooth_loss(self, disp, img):
        b, _, h, w = disp.size()
        a1 = 0.5
        a2 = 0.5
        img = F.interpolate(img, (h, w), mode='area')

        disp_dx, disp_dy = self.gradient(disp)
        img_dx, img_dy = self.gradient(img)

        disp_dxx, disp_dxy = self.gradient(disp_dx)
        disp_dyx, disp_dyy = self.gradient(disp_dy)

        img_dxx, img_dxy = self.gradient(img_dx)
        img_dyx, img_dyy = self.gradient(img_dy)

        smooth1 = torch.mean(disp_dx.abs() * torch.exp(-a1 * img_dx.abs().mean(1, True))) + \
                  torch.mean(disp_dy.abs() * torch.exp(-a1 * img_dy.abs().mean(1, True)))

        smooth2 = torch.mean(disp_dxx.abs() * torch.exp(-a2 * img_dxx.abs().mean(1, True))) + \
                  torch.mean(disp_dxy.abs() * torch.exp(-a2 * img_dxy.abs().mean(1, True))) + \
                  torch.mean(disp_dyx.abs() * torch.exp(-a2 * img_dyx.abs().mean(1, True))) + \
                  torch.mean(disp_dyy.abs() * torch.exp(-a2 * img_dyy.abs().mean(1, True)))

        return smooth1 + smooth2

    def gradient(self, D):
        D_dy = D[:, :, 1:] - D[:, :, :-1]
        D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return D_dx, D_dy
