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
from thop import clever_format
from thop import profile
import torchvision
from torchvision import transforms
from thop import clever_format
from thop import profile
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
        # print("inputs keys--------------",inputs.keys())
        # print("color shape************", inputs[("color", 0, 0)].shape)
        # print("both shape************", inputs[("both", 0, 0)].shape)
  #      start = time.time()
        depth_feature = self.DepthEncoder(inputs["color_aug", 0, 0])
        print("flops and params of DepthEncoder")
        flops, params = profile(self.DepthEncoder, inputs=(inputs["color_aug", 0, 0],))
        flops_DepthEncoder, params_DepthEncoder= clever_format([flops, params], "%.3f")
        print(flops_DepthEncoder, params_DepthEncoder)

   #     end = time.time()
 #       print("depth encoder time:", end - start)

        outputs = self.DepthDecoder(depth_feature)
        print("flops and params of DepthDecoder")
        flops, params = profile(self.DepthDecoder, inputs=(depth_feature,))
        flops_DepthDecoder, params_DepthDecoder= clever_format([flops, params], "%.3f")
        print(flops_DepthDecoder, params_DepthDecoder)

 #       end2 = time.time()
 #       print("depth decoder time:", end2 - end)
        layout, feature = self.predict_layout(inputs, depth_feature)
        outputs.update(layout)
        encoder_features = feature
        outputs.update(self.predict_layoutB(inputs, depth_feature, encoder_features))
 #       end3 = time.time()
 #       print("predict layout time:", end3 - end2)

        outputs.update(self.predict_poses(inputs))
 #            end4 = time.time()
 #           print("predict pose time:", end4 - end3)
            # print("outputs", outputs.keys())
        if self.training:
            loss_dict = self.compute_losses(inputs, outputs)
 #           end5 = time.time()
#            print("compute loss time:", end5 - end4)
 #           print("one batch time:", end5 - start)
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
        # print("topview*****************",outputs["topview"].shape)
        # print("target*****************", inputs["both",0,0].shape)
        #print(self.opt["type"])
        loss_dict = {}
  #      start2 = time.time()
  #       if self.opt["type"] == "both":
  #           weight = torch.Tensor([1., self.weight["dynamic"], self.weight["static"]])
        if self.opt["type"] == "static_raw" or self.opt["type"] == "static" or self.opt["type"] == "Argo_static" or self.opt["type"] == "Argo_both":

            weightS = torch.Tensor([1., self.weight["static"]])
        if self.opt["type"] == "dynamic" or self.opt["type"] == "Argo_dynamic" or self.opt["type"] == "Argo_both":
            weightD = torch.Tensor([1., self.weight["dynamic"]])
        if self.opt["type"] == "static" or self.opt["type"] == "static_raw" or self.opt["type"] == "Argo_static" or self.opt["type"] == "Argo_both":
            scale_label = self.get_scale_label_static(inputs, self.opt)
        elif self.opt["type"] == "dynamic" or self.opt["type"] == "Argo_dynamic":
            scale_label = self.get_scale_label_dynamic(inputs, self.opt)
        #endl1 = time.time()
  #      print("compute scale label time:", endl1 - start2)
        loss_dict["topview_loss"] = 0
        loss_dict["transform_topview_loss"] = 0
        loss_dict["transform_loss"] = 0
        loss_dict["topview_lossB"] = 0
        loss_dict["transform_topview_lossB"] = 0
        loss_dict["transform_lossB"] = 0
        # print("outputs[topview]:", outputs["topview"].shape)
        # print("inputs[both]:", inputs["both",0,0].shape)
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
       # endl2 = time.time()
  #      print("compute topview loss time:", endl2 - endl1)
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
  #          endl3 = time.time()
  #          print("compute photometric loss time: "+str(scale), endl3 - start3)
            scale_loss = self.get_scale_loss(inputs, outputs, scale, scale_label)
            loss_dict[('scale_loss', scale)] = self.opt.scale_weight * scale_loss / (2 ** scale) / len(
                self.opt.scales)
  #          endl4 = time.time()
  #          print("compute scale loss time: "+str(scale), endl4 - endl3)
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
   #         endl5 = time.time()
   #         print("compute smooth loss time: " + str(scale), endl5 - endl4)

        return loss_dict
    def get_scale_loss(self, inputs, outputs, scale, scale_label):
        depth_pred = outputs[("depth", 0, scale)]
        # scale_label = self.get_scale_label(inputs)
        shape = scale_label.shape[2:4]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, shape, mode="bilinear", align_corners=False), 1e-3, 80)
        # print("depth_pred",depth_pred.shape)
        # depth_pred = depth_pred.detach()

        depth_gt = scale_label
        mask = depth_gt > 0
        # print("mask", mask.shape)
        if self.opt["type"] == "static_raw":# or self.opt["type"] == "static":
        # garg/eigen crop
            crop_mask = torch.zeros_like(mask)
            crop_mask[:, :, 153:371, 44:1197] = 1
            mask = mask * crop_mask
        # print("mask", mask.shape)
        depth_gt = torch.masked_select(depth_gt, mask)
        depth_pred = torch.masked_select(depth_pred, mask)
        # depth_pred = depth_pred[mask]
        abs_rel_loss = torch.mean(torch.abs(depth_gt - depth_pred) / depth_gt)
        return abs_rel_loss
    def get_scale_label_static(self, inputs, opt):
        # sequence = str(inputs[("bev_path", 0, 0)]["sequence"])
        # print("sequence: ", sequence)
        # frame_index = str(inputs[("bev_path", 0, 0)]["frame_index"])
        # print("frame_index: ", frame_index)
        # basedir = '/public/data1/users/zhaohaimei3/kitti_data/odometry/dataset'
        # frame_path = os.path.join(basedir, "sequences", sequence, "image_2")
        # frame_path = os.path.join(frame_path, "%06d.png" % int(frame_index))
        # bev_path = frame_index.replace("image_2", "road_dense128")
        # data = inputs[("odometry_data", 0, 0)]
        # img_front = imageio.imread(frame_path)
        img_front = inputs[("color", 0, -1)]
        mapsize = opt.occ_map_size
        # img_front shape [128, 128, 3]
        # img batch [8, 3, 128, 128]


        # fig = plt.figure(figsize=(15, 10))
        # plt.imsave("./" + str(0) + "_frontImg.png", img_front[0])
        # print("img_front------------------", img_front.shape)
        height, width = img_front.shape[2:4]
        # img_front_save = np.array(img_front[0].resize(height,width,3).cpu())
        fig = plt.figure(figsize=(15, 10))
        # plt.imsave("./frontImg.png", np.array(img_front[0][0].cpu()))
        # bev_img_layout = imageio.imread(bev_path)
        # bev_img_layout = inputs[("dynamic", 0, 0)]
        # plt.imsave("./dynamic.png", np.array(bev_img_layout[0][0].cpu()))
        bev_img_layout = inputs[("bothS", 0, 0)]
        # plt.imsave("./both_dynamic.png", np.array(bev_img_layout[0][0].cpu()))

        # bev_layout_distance_label = torch.zeros(bev_img_layout.shape).cuda()
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
        # yr = torch.arange(0, int(mapsize / 2), step=1).view(1, 1, 1, int(w / 2)).repeat(batchsize, 1, h, 1) * (
        #             40 / mapsize)
        # yl = torch.arange(int(mapsize / 2), 0, step=-1).view(1, 1, 1, int(w / 2)).repeat(batchsize, 1, h, 1) * (
        #             40 / mapsize)
        # y = torch.cat([yl, yr], 3)
        # d = torch.sqrt(z ** 2 + y ** 2)
        bev_layout_distance_label = z.cuda()
        # for i in range(bev_layout_distance_label.shape[2]):
        #     bev_layout_distance_label[:, :, i, :] = (256 - i) * (40 / 256) - 1.9
        points_real = [(round(18 * resolution1), round(31 * resolution1)),
                       (round(22 * resolution1), round(31 * resolution1)),
                       (round(18 * resolution1), round(33 * resolution1)),
                       (round(22 * resolution1), round(33 * resolution1))]
        # points_real = [(58, 99),
        #                (70, 99),
        #                (58, 106),
        #                (70, 106)]
        bev_img_layout = torch.fliplr(bev_img_layout)
        bev_layout_distance_label = torch.fliplr(bev_layout_distance_label)
        bev_img_layout = torchvision.transforms.functional.rotate(bev_img_layout, angle=270)
        # plt.imsave("./rotated_layout.png", bev_img_layout)
        bev_layout_distance_label = torchvision.transforms.functional.rotate(bev_layout_distance_label, angle=270)
        # points_real_rot = [[256 - points_real[3][1] - 1, points_real[0][0] - 1, 1],
        #                    [256 - points_real[3][1] + (points_real[2][1] - points_real[1][1]) - 1,
        #                     points_real[0][0] - 1, 1],
        #                    [256 - points_real[3][1] - 1, points_real[1][0] - 1, 1],
        #                    [256 - points_real[3][1] + (points_real[2][1] - points_real[1][1]) - 1,
        #                     points_real[1][0] - 1, 1]]
        points_real_rot = [[mapsize - points_real[3][1] - 1, points_real[0][0] - 1],
                           [mapsize - points_real[3][1] + (points_real[2][1] - points_real[1][1]) - 1,
                            points_real[0][0] - 1],
                           [mapsize - points_real[3][1] - 1, points_real[1][0] - 1],
                           [mapsize - points_real[3][1] + (points_real[2][1] - points_real[1][1]) - 1,
                            points_real[1][0] - 1]]
        # K = data.calib.K_cam2
        # K_np = inputs[("odometry_K", 0, 0)].cpu()
        # T_K_np = K_np.T
        # print("T_K_np^^^^^^^^^^^^^^^^^^",T_K_np)
        K = inputs[("odometry_K", 0, 0)][:,:3,:3]
        batchsize = K.shape[0]
        # print("K^^^^^^^^^^^^^^^^^^", K.shape)
        # print("K^^^^^^^^^^^^^^^^^^", K)
        # T_K_tensor = K.transpose(0,2)
        # print("T_K_tensor^^^^^^^^^^^^^^^^^^", T_K_tensor)
        # T_K_tensor1 = K.transpose(0, 1)
        # print("T_K_tensor1^^^^^^^^^^^^^^^^^^", T_K_tensor1)
        # Tr_cam2_velo = data.calib.T_cam2_velo  # to cam2 frame from velodyne frame
        Tr_cam2_velo = inputs[("Tr_cam2_velo", 0, 0)]
        # print("Tr_cam2_velo shape", type(Tr_cam2_velo))
        camera_SE3_egovehicleO = Tr_cam2_velo
        camera_R_egovehicle = camera_SE3_egovehicleO[:, :3, :3]#.cpu()
        # print("camera_R_egovehicle", camera_R_egovehicle)
        # print("camera_R_egovehicle",camera_R_egovehicle.shape)

        camera_R_egovehicle_Transpose = camera_R_egovehicle.transpose(1, 2)
        # print("camera_R_egovehicleTranspose", camera_R_egovehicle_Transpose.shape)
        # print("camera_R_egovehicleTranspose", camera_R_egovehicle_Transpose)
        camera_t_egovehicle = camera_SE3_egovehicleO[:, :3, 3]#.cpu()
        # print("camera_t_egovehicle", camera_t_egovehicle.shape)
        # camera_t_egovehicleT = camera_t_egovehicle.cpu().T
        # print("camera_t_egovehicleT", camera_t_egovehicleT)
        # camera_t_egovehicleTensor = camera_t_egovehicle.transpose(0,1)
        # print("camera_t_egovehicleTensor", camera_t_egovehicleTensor)
        # camera_SE3_egovehicle_list=[]
        # for i in range(camera_R_egovehicle.shape[0]):
        camera_SE3_egovehicle = SE3(rotation=camera_R_egovehicle, translation=camera_t_egovehicle)
        # print("camera_SE3_egovehicle transform_matrix",camera_SE3_egovehicle.transform_matrix)

        if opt.split == "argo":
            HEIGHT_FROM_GROUND = 0.33
        elif opt.split == "raw" or opt.split == "odometry":
            HEIGHT_FROM_GROUND = 1.73  # in meters,
        ground_rotation = torch.eye(3).repeat(batchsize,1,1)
        # print("ground_rotation",ground_rotation.shape)
        ground_translation = torch.Tensor([0, 0, HEIGHT_FROM_GROUND]).repeat(batchsize,1)
        # print("ground_translation", ground_translation.shape)
        ground_SE3_egovehicle = SE3(rotation=ground_rotation, translation=ground_translation)
        egovehicle_SE3_ground = ground_SE3_egovehicle(None, "inverse")
        egovehicle_SE3_ground_list = []
        # for i in range(camera_R_egovehicle.shape[0]):
        # egovehicle_SE3_ground=egovehicle_SE3_ground
        # camera_SE3_ground_list = []
        # img_H_ground_list =[]
        # for i in range(camera_R_egovehicle.shape[0]):
        camera_SE3_ground = camera_SE3_egovehicle(egovehicle_SE3_ground,"right_multiply_with_se3")
        # print("camera_SE3_ground&&&&&&&&&&&&&&&&&&&&&&&&", camera_SE3_ground.transform_matrix)
        img_H_ground=self.homography_from_calibration(camera_SE3_ground, K)
        # print("img_H_ground&&&&&&&&&&&&&&&&&&&&&&&&", img_H_ground)
        ground_H_img = torch.linalg.inv(img_H_ground)
        # print("ground_H_img", ground_H_img.shape)
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
        # print("shiftedground_H_ground",shiftedground_H_ground.shape)
        # print("ground_H_img***********************", ground_H_img[0])
        # print("shiftedground_H_ground***********************", shiftedground_H_ground[0])
        shiftedground_H_img = torch.bmm(shiftedground_H_ground, ground_H_img)
        #print("***********************", torch.linalg.inv(shiftedground_H_img)[0])
        
        # restore_front_layout = cv2.warpPerspective(bev_img_layout, np.linalg.inv(shiftedground_H_img),
        #                                            dsize=(width, height))
        #plt.imsave("./rotated_layout2.png", np.array(bev_img_layout[0][0].cpu()))
        restore_front_layout = warp_perspective(bev_img_layout, torch.linalg.inv(shiftedground_H_img),
                                                   dsize=(height, width))
        # restore_front_bev_layout_distance_label = cv2.warpPerspective(bev_layout_distance_label,
        #                                                               np.linalg.inv(shiftedground_H_img),
        #                                                               dsize=(width, height))
        restore_front_bev_layout_distance_label = warp_perspective(bev_layout_distance_label,
                                                                      torch.linalg.inv(shiftedground_H_img),
                                                                      dsize=(height, width))
        # for i in range(restore_front_bev_layout_distance_label.shape[2]):
        #     # if i < 320 and i > 280:
        #     # print(np.max(restore_front_bev_layout_distance_label[i,:]))
        #     print(np.max(np.array(restore_front_bev_layout_distance_label[0, 0, i, :].cpu())))
        layout_mask = restore_front_layout
        #plt.imsave("./layout_mask.png", np.array(layout_mask[0][0].cpu()))
        points_real_rot = np.asarray(points_real_rot)

        #
        # print("!"*80)
        # print(points_real_rot)
        # print(type(points_real_rot))
        # print("!"*80)
        points_real_rot = torch.tensor(points_real_rot,dtype=torch.float32).repeat(batchsize,1,1).cuda()
        # print("points_real_rot shape", points_real_rot.shape)
            # .repeat(batchsize,1,1).cuda()#(B,4,3)
        newPointsO = torch.round(transform_points(torch.linalg.inv(shiftedground_H_img), points_real_rot)).int()#(B,4,3)
        # print("newPoints-------", newPointsO.shape)
        # print("newPoints-------", newPointsO)
        newPoints = newPointsO.cpu()
        pts = np.array(
            [[newPoints[0][0][0], newPoints[0][0][1]], [newPoints[0][2][0], newPoints[0][2][1]], [newPoints[0][3][0], newPoints[0][3][1]],
             [newPoints[0][1][0], newPoints[0][1][1]]])
        pts = np.round(pts).astype(np.int32)
        # # print("bts before reshape", pts)
        pts = pts.reshape((-1, 1, 2))
        # print("bts after reshape", pts)
        imgshape = [img_front.shape[2:4][0], img_front.shape[2:4][1], 3]
        # # print("imgshape---------",imgshape)
        img_zero = np.zeros(imgshape, dtype=np.uint8)
        # # print("img_zero reshape", img_zero.shape)
        restore_front_layout = cv2.fillConvexPoly(img_zero, pts, (0, 255, 255), 1)
        # # print("restore_front_layout reshape", restore_front_layout.shape)
        img_gray_roi_mask_triangle = cv2.cvtColor(restore_front_layout, cv2.COLOR_RGB2GRAY)
        img_gray_roi_mask_triangle[img_gray_roi_mask_triangle > 0] = 255
        # cv2.imwrite('img_gray_roi_mask_triangle.png', img_gray_roi_mask_triangle)
        img_gray_roi_mask_triangle = torch.tensor(img_gray_roi_mask_triangle).repeat(batchsize,1,1).unsqueeze(1).cuda()
        # # print("img_gray_roi_mask_triangle shape", img_gray_roi_mask_triangle.shape)
        # # print("img_gray_roi_mask_triangle dtype", img_gray_roi_mask_triangle.dtype)
        # # print("layout_mask shape", layout_mask.shape)
        # # print("layout_mask dtype", layout_mask.dtype)
        layout_mask = layout_mask.type_as(img_gray_roi_mask_triangle)
        # plt.imsave("./layout_mask.png", np.array(layout_mask[0][0].cpu()))
        # # for i in range(img_gray_roi_mask_triangle.shape[2]):
        # #     print(np.array(img_gray_roi_mask_triangle[0, 0, i,:].cpu()))
        A_and_B = torch.bitwise_and(layout_mask, img_gray_roi_mask_triangle)
        # for i in range(A_and_B.shape[2]):
        #     print(np.array(A_and_B[0, 0, i,:].cpu()))
        # A_and_B[A_and_B < 128] = 0
        # A_and_B[A_and_B > 127] = 1
        # plt.imsave("./_A_and_B_255.png", np.array(A_and_B[0][0].cpu()))
        restore_front_bev_layout_distance_label = restore_front_bev_layout_distance_label * A_and_B
        # for i in range(restore_front_bev_layout_distance_label.shape[2]):
        #     # if i < 320 and i > 280:
        #     # print(np.max(restore_front_bev_layout_distance_label[i,:]))
        #     print(np.max(np.array(restore_front_bev_layout_distance_label[0, 0, i,:].cpu())))
        # print("restore_front_bev_layout_distance_label shape", restore_front_bev_layout_distance_label.shape)
        # pred_disp = restore_front_bev_layout_distance_label[0][0].cpu()
        # img_path = os.path.join("./depth_label_map.png")
        # vmax = np.percentile(pred_disp, 95)
        #
        # plt.imsave(img_path, pred_disp, cmap='magma', vmax=vmax)
        return restore_front_bev_layout_distance_label
    def get_scale_label_dynamic(self, inputs, opt):
        # sequence = str(inputs[("bev_path", 0, 0)]["sequence"])
        # print("sequence: ", sequence)
        # frame_index = str(inputs[("bev_path", 0, 0)]["frame_index"])
        # print("frame_index: ", frame_index)
        # basedir = '/public/data1/users/zhaohaimei3/kitti_data/odometry/dataset'
        # frame_path = os.path.join(basedir, "sequences", sequence, "image_2")
        # frame_path = os.path.join(frame_path, "%06d.png" % int(frame_index))
        # bev_path = frame_index.replace("image_2", "road_dense128")
        # data = inputs[("odometry_data", 0, 0)]
        # img_front = imageio.imread(frame_path)
        img_front = inputs[("color", 0, -1)]
        # img_front shape [128, 128, 3]
        # img batch [8, 3, 128, 128]
        mapsize = opt.occ_map_size

        # fig = plt.figure(figsize=(15, 10))
        # plt.imsave("./" + str(0) + "_frontImg.png", img_front[0])
        # print("img_front------------------", img_front.shape)
        height, width = img_front.shape[2:4]
        # img_front_save = np.array(img_front[0].resize(height,width,3).cpu())
        # fig = plt.figure(figsize=(15, 10))
        # plt.imsave("./frontImg.png", img_front_save)
        # bev_img_layout = imageio.imread(bev_path)
        bev_img_layout = inputs[("bothS", 0, 0)]
        # print("bev_img_layout------------------", bev_img_layout.shape)
        # bev_layout_distance_label = torch.zeros(bev_img_layout.shape).cuda()
        resolution = 40 / mapsize
        resolution1 = mapsize / 40
        if bev_img_layout.shape[2]!= mapsize or bev_img_layout.shape[3]!= mapsize:
            raise ValueError("The shape of both label is not ", mapsize)
        batchsize = bev_img_layout.shape[0]
        h = w = mapsize
        # bev_layout_distance_label = torch.arange(mapsize, 0, step=-1). \
        #                                 view(1, 1, h, 1).repeat(batchsize, 1, 1, w) * (40 / mapsize) #- 0.27
        # bev_layout_distance_label = bev_layout_distance_label.cuda()
        if opt.split == "argo":
            z = torch.arange(mapsize, 0, step=-1).view(1, 1, h, 1).repeat(batchsize, 1, 1, w) * (40 / mapsize) - 1.9
        elif opt.split == "raw" or "odometry":
            z = torch.arange(mapsize, 0, step=-1).view(1, 1, h, 1).repeat(batchsize, 1, 1, w) * (40 / mapsize)  # - 0.27
        # yr = torch.arange(0, int(mapsize / 2), step=1).view(1, 1, 1, int(w / 2)).repeat(batchsize, 1, h, 1) * (
        #         40 / mapsize)
        # yl = torch.arange(int(mapsize / 2), 0, step=-1).view(1, 1, 1, int(w / 2)).repeat(batchsize, 1, h, 1) * (
        #         40 / mapsize)
        # y = torch.cat([yl, yr], 3)
        # d = torch.sqrt(z ** 2 + y ** 2)
        bev_layout_distance_label = z.cuda()
        # for i in range(bev_layout_distance_label.shape[2]):
        #     bev_layout_distance_label[:, :, i, :] = (256 - i) * (40 / 256) - 0.27
        points_real = [(round(18 * resolution1), round(31 * resolution1)),
                  (round(22 * resolution1), round(31 * resolution1)),
                  (round(18 * resolution1), round(33 * resolution1)),
                  (round(22 * resolution1), round(33 * resolution1))]
        # points_real = [(58, 99),
        #                (70, 99),
        #                (58, 106),
        #                (70, 106)]
        # bev_img_layout = torch.fliplr(bev_img_layout)
        bev_layout_distance_label = torch.fliplr(bev_layout_distance_label)
        # bev_img_layout = torchvision.transforms.functional.rotate(bev_img_layout, angle=270)
        bev_layout_distance_label = torchvision.transforms.functional.rotate(bev_layout_distance_label, angle=270)
        # points_real_rot = [[256 - points_real[3][1] - 1, points_real[0][0] - 1, 1],
        #                    [256 - points_real[3][1] + (points_real[2][1] - points_real[1][1]) - 1,
        #                     points_real[0][0] - 1, 1],
        #                    [256 - points_real[3][1] - 1, points_real[1][0] - 1, 1],
        #                    [256 - points_real[3][1] + (points_real[2][1] - points_real[1][1]) - 1,
        #                     points_real[1][0] - 1, 1]]
        points_real_rot = [[mapsize - points_real[3][1] - 1, points_real[0][0] - 1],
                           [mapsize - points_real[3][1] + (points_real[2][1] - points_real[1][1]) - 1,
                            points_real[0][0] - 1],
                           [mapsize - points_real[3][1] - 1, points_real[1][0] - 1],
                           [mapsize - points_real[3][1] + (points_real[2][1] - points_real[1][1]) - 1,
                            points_real[1][0] - 1]]
        # K = data.calib.K_cam2
        # K_np = inputs[("odometry_K", 0, 0)].cpu()
        # T_K_np = K_np.T
        # print("T_K_np^^^^^^^^^^^^^^^^^^",T_K_np)
        K = inputs[("odometry_K", 0, 0)][:, :3, :3]
        batchsize = K.shape[0]
        # print("K^^^^^^^^^^^^^^^^^^", K.shape)
        # print("K^^^^^^^^^^^^^^^^^^", K)
        # T_K_tensor = K.transpose(0,2)
        # print("T_K_tensor^^^^^^^^^^^^^^^^^^", T_K_tensor)
        # T_K_tensor1 = K.transpose(0, 1)
        # print("T_K_tensor1^^^^^^^^^^^^^^^^^^", T_K_tensor1)
        # Tr_cam2_velo = data.calib.T_cam2_velo  # to cam2 frame from velodyne frame
        Tr_cam2_velo = inputs[("Tr_cam2_velo", 0, 0)]
        # print("Tr_cam2_velo shape", type(Tr_cam2_velo))
        camera_SE3_egovehicleO = Tr_cam2_velo
        camera_R_egovehicle = camera_SE3_egovehicleO[:, :3, :3]#.cpu()
        # print("camera_R_egovehicle", camera_R_egovehicle)
        # print("camera_R_egovehicle",camera_R_egovehicle.shape)

        camera_R_egovehicle_Transpose = camera_R_egovehicle.transpose(1, 2)
        # print("camera_R_egovehicleTranspose", camera_R_egovehicle_Transpose.shape)
        # print("camera_R_egovehicleTranspose", camera_R_egovehicle_Transpose)
        camera_t_egovehicle = camera_SE3_egovehicleO[:, :3, 3]#.cpu()
        # print("camera_t_egovehicle", camera_t_egovehicle.shape)
        # camera_t_egovehicleT = camera_t_egovehicle.cpu().T
        # print("camera_t_egovehicleT", camera_t_egovehicleT)
        # camera_t_egovehicleTensor = camera_t_egovehicle.transpose(0,1)
        # print("camera_t_egovehicleTensor", camera_t_egovehicleTensor)
        # camera_SE3_egovehicle_list=[]
        # for i in range(camera_R_egovehicle.shape[0]):
        camera_SE3_egovehicle = SE3(rotation=camera_R_egovehicle, translation=camera_t_egovehicle)
        # print("camera_SE3_egovehicle transform_matrix",camera_SE3_egovehicle.transform_matrix)

        if opt.split == "argo":
            HEIGHT_FROM_GROUND = 0.33
        elif opt.split == "raw" or "odometry":
            HEIGHT_FROM_GROUND = 1.73  # in meters,
        ground_rotation = torch.eye(3).repeat(batchsize,1,1)
        # print("ground_rotation",ground_rotation.shape)
        ground_translation = torch.Tensor([0, 0, HEIGHT_FROM_GROUND]).repeat(batchsize,1)
        # print("ground_translation", ground_translation.shape)
        ground_SE3_egovehicle = SE3(rotation=ground_rotation, translation=ground_translation)
        egovehicle_SE3_ground = ground_SE3_egovehicle(None, "inverse")
        egovehicle_SE3_ground_list = []
        # for i in range(camera_R_egovehicle.shape[0]):
        # egovehicle_SE3_ground=egovehicle_SE3_ground
        # camera_SE3_ground_list = []
        # img_H_ground_list =[]
        # for i in range(camera_R_egovehicle.shape[0]):
        camera_SE3_ground = camera_SE3_egovehicle(egovehicle_SE3_ground,"right_multiply_with_se3")
        # print("camera_SE3_ground&&&&&&&&&&&&&&&&&&&&&&&&", camera_SE3_ground.transform_matrix)
        img_H_ground=self.homography_from_calibration(camera_SE3_ground, K)
        # print("img_H_ground&&&&&&&&&&&&&&&&&&&&&&&&", img_H_ground)
        ground_H_img = torch.linalg.inv(img_H_ground)
        # print("ground_H_img", ground_H_img.shape)
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
        # print("shiftedground_H_ground",shiftedground_H_ground.shape)
        # print("ground_H_img***********************", ground_H_img[0])
        # print("shiftedground_H_ground***********************", shiftedground_H_ground[0])
        shiftedground_H_img = torch.bmm(shiftedground_H_ground, ground_H_img)
        # print("***********************", shiftedground_H_img[0])
        # restore_front_layout = cv2.warpPerspective(bev_img_layout, np.linalg.inv(shiftedground_H_img),
        #                                            dsize=(width, height))
        # restore_front_layout = warp_perspective(bev_img_layout, torch.linalg.inv(shiftedground_H_img),
        #                                            dsize=(height, width))
        # restore_front_bev_layout_distance_label = cv2.warpPerspective(bev_layout_distance_label,
        #                                                               np.linalg.inv(shiftedground_H_img),
        #                                                               dsize=(width, height))
        restore_front_bev_layout_distance_label = warp_perspective(bev_layout_distance_label,
                                                                      torch.linalg.inv(shiftedground_H_img),
                                                                      dsize=(height, width))
        # for i in range(restore_front_bev_layout_distance_label.shape[2]):
        #     # if i < 320 and i > 280:
        #     # print(np.max(restore_front_bev_layout_distance_label[i,:]))
        #     print(np.max(np.array(restore_front_bev_layout_distance_label[0, 0, i, :].cpu())))
        # layout_mask = restore_front_layout

        points_real_rot = np.asarray(points_real_rot)

        #
        # print("!"*80)
        # print(points_real_rot)
        # print(type(points_real_rot))
        # print("!"*80)
        points_real_rot = torch.tensor(points_real_rot,dtype=torch.float32).repeat(batchsize,1,1).cuda()
        # print("points_real_rot shape", points_real_rot.shape)
            # .repeat(batchsize,1,1).cuda()#(B,4,3)
        newPointsO = torch.round(transform_points(torch.linalg.inv(shiftedground_H_img), points_real_rot)).int()#(B,4,3)
        # print("newPoints-------", newPointsO.shape)
        # print("newPoints-------", newPointsO)
        newPoints = newPointsO.cpu()
        pts = np.array(
            [[newPoints[0][0][0], newPoints[0][0][1]], [newPoints[0][2][0], newPoints[0][2][1]], [newPoints[0][3][0], newPoints[0][3][1]],
             [newPoints[0][1][0], newPoints[0][1][1]]])
        pts = np.round(pts).astype(np.int32)
        # print("bts before reshape", pts)
        pts = pts.reshape((-1, 1, 2))
        # print("bts after reshape", pts)
        imgshape = [img_front.shape[2:4][0], img_front.shape[2:4][1], 3]
        # print("imgshape---------",imgshape)
        img_zero = np.zeros(imgshape, dtype=np.uint8)
        # print("img_zero reshape", img_zero.shape)
        restore_front_layout = cv2.fillConvexPoly(img_zero, pts, (0, 255, 255), 1)
        # print("restore_front_layout reshape", restore_front_layout.shape)
        img_gray_roi_mask_triangle = cv2.cvtColor(restore_front_layout, cv2.COLOR_RGB2GRAY)
        img_gray_roi_mask_triangle[img_gray_roi_mask_triangle > 0] = 1
        # cv2.imwrite('img_gray_roi_mask_triangle.png', img_gray_roi_mask_triangle*255)
        img_gray_roi_mask_triangle = torch.tensor(img_gray_roi_mask_triangle).repeat(batchsize,1,1).unsqueeze(1).cuda()
        # print("img_gray_roi_mask_triangle shape", img_gray_roi_mask_triangle.shape)
        # print("img_gray_roi_mask_triangle dtype", img_gray_roi_mask_triangle.dtype)
        # print("layout_mask shape", layout_mask.shape)
        # print("layout_mask dtype", layout_mask.dtype)
        # layout_mask = layout_mask.type_as(img_gray_roi_mask_triangle)
        # plt.imsave("./layout_mask.png", np.array(layout_mask[0][0].cpu()))
        # for i in range(img_gray_roi_mask_triangle.shape[2]):
        #     print(np.array(img_gray_roi_mask_triangle[0, 0, i,:].cpu()))
        # A_and_B = torch.bitwise_and(layout_mask, img_gray_roi_mask_triangle)
        # for i in range(A_and_B.shape[2]):
        #     print(np.array(A_and_B[0, 0, i,:].cpu()))
        # A_and_B[A_and_B < 128] = 0
        # A_and_B[A_and_B > 127] = 1
        # plt.imsave("./_A_and_B_255.png", np.array(A_and_B[0][0].cpu()))
        restore_front_bev_layout_distance_label = restore_front_bev_layout_distance_label * img_gray_roi_mask_triangle
        # for i in range(restore_front_bev_layout_distance_label.shape[2]):
        #     # if i < 320 and i > 280:
        #     # print(np.max(restore_front_bev_layout_distance_label[i,:]))
        #     print(np.max(np.array(restore_front_bev_layout_distance_label[0, 0, i,:].cpu())))
        # print("restore_front_bev_layout_distance_label shape", restore_front_bev_layout_distance_label.shape)
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
                pose_inputs_ = self.PoseEncoder(torch.cat(pose_inputs, 1))
                flops, params = profile(self.PoseEncoder, inputs=(torch.cat(pose_inputs, 1),))
                flops_encoder, params_encoder = clever_format([flops, params], "%.3f")
                print("poseEnco"+str(f_i)+": ",flops_encoder, params_encoder)
                axisangle, translation = self.PoseDecoder(pose_inputs_)
                flops, params = profile(self.PoseDecoder, inputs=(pose_inputs_,))
                flops_decoder, params_decoder = clever_format([flops, params], "%.3f")           
                print("poseDeco"+str(f_i)+": ",flops_decoder, params_decoder)
                outputs[("cam_T_cam", 0, f_i)] = self.transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        return outputs

    def predict_layout(self, inputs, depth_feature):
        outputs = {}
        features = self.LayoutEncoder(inputs["color_aug", 0, 0])
        print("flops and params of encoder")
        flops, params = profile(self.LayoutEncoder, inputs=(inputs["color",0,0],))
        flops_encoder, params_encoder= clever_format([flops, params], "%.3f")
        print(flops_encoder, params_encoder)
        encoder_features = features
        # Cross-view Transformation Module

        transform_feature, retransform_features = self.CycledViewProjection(features)
        # print("transform_feature#####################",transform_feature.shape)
        # print("retransform_features#####################", retransform_features.shape)
        depth_feature = depth_feature[-1]
        # print("depth_feature#####################", depth_feature.shape)
        features = self.CrossViewTransformer(features, transform_feature, retransform_features,depth_feature)

        outputs["topview"] = self.LayoutDecoder(features)
        print("flops and params of decoder")
        flops, params = profile(self.LayoutDecoder, inputs=(features,))
        flops_decoder, params_decoder= clever_format([flops, params], "%.3f")
        print(flops_decoder, params_decoder)
        outputs["transform_topview"] = self.LayoutTransformDecoder(transform_feature)
        print("flops and params of transform_decoder")
        flops, params = profile(self.LayoutTransformDecoder, inputs=(transform_feature,))
        flops_transform_feature, params_transform_feature= clever_format([flops, params], "%.3f")
        print(flops_transform_feature, params_transform_feature)
        outputs["features"] = features
        outputs["retransform_features"] = retransform_features
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
        features = self.CrossViewTransformerB(features, transform_feature, retransform_features,depth_feature)

        outputs["topviewB"] = self.LayoutDecoderB(features)
        print("flops and params of decoderB")                                            
        flops, params = profile(self.LayoutDecoderB, inputs=(features,))                 
        flops_decoder, params_decoder= clever_format([flops, params], "%.3f")            
        print(flops_decoder, params_decoder)
        outputs["transform_topviewB"] = self.LayoutTransformDecoderB(transform_feature)
        print("flops and params of transform_decoderB")
        flops, params = profile(self.LayoutTransformDecoderB, inputs=(transform_feature,))
        flops_transform_feature, params_transform_feature= clever_format([flops, params], "%.3f")
        print(flops_transform_feature, params_transform_feature)
        outputs["featuresB"] = features
        outputs["retransform_featuresB"] = retransform_features
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
