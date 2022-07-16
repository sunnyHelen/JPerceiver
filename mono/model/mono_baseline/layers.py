from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, norm_layer):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode='nearest')
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode='nearest')
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode='nearest')
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode='nearest')
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)


def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1 / max_depth #0.01
    max_disp = 1 / min_depth #10
    scaled_disp = min_disp + (max_disp - min_disp) * disp #(10-0.01)*disp+0.01
    depth = 1 / scaled_disp
    return scaled_disp, depth


class Backproject(nn.Module):
    def __init__(self, batch_size, height, width):
        super(Backproject, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = torch.from_numpy(self.id_coords)
        self.ones = torch.ones(self.batch_size, 1, self.height * self.width)
        self.pix_coords = torch.unsqueeze(torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = torch.cat([self.pix_coords, self.ones], 1)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords.cuda())
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones.cuda()], 1)
        return cam_points


class Project(nn.Module):
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def upsample(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


def upshuffle(in_planes, upscale_factor):
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, in_planes*upscale_factor**2, kernel_size=3, stride=1, padding=0),
        nn.PixelShuffle(upscale_factor),
        nn.ELU(inplace=True)
    )


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.pad = nn.ReflectionPad2d((0, 1, 0, 1))
        self.nonlin = nn.ELU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.pad(out)
        out = self.nonlin(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=1, bias=bias)
    def forward(self, x):
        out = self.conv(x)
        return out


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class Conv5x5(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv5x5, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(2)
        else:
            self.pad = nn.ZeroPad2d(2)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 5)
    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'pointwise'), Conv1x1(in_planes if (i == 0) else out_planes, out_planes, False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'pointwise'))(top)
            x = top + x
        return x


def compute_depth_errors(gt, pred):
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())
    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean((gt - pred) ** 2 / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
class SE3(nn.Module):

    """An SE3 class allows point cloud rotation and translation operations."""

    def __init__(self, rotation: torch.Tensor, translation: torch.Tensor) -> None:
        super(SE3, self).__init__()
        """Initialize an SE3 instance with its rotation and translation matrices.

        Args:
            rotation: Tensor of shape (B, 3, 3)
            translation: Tensor of shape (B,3,)
        """
        # print("rotation[0].shape",rotation[0].shape)
        # print("translation[0].shape", translation[0].shape)
        assert rotation[0].shape == (3, 3)
        assert translation[0].shape == (3,)
        self.rotation = rotation
        self.translation = translation
        batch = self.rotation.shape[0]
        self.transform_matrix = torch.eye(4).repeat(batch, 1, 1).cuda()
        # print("self.transform_matrix", self.transform_matrix)
        self.transform_matrix[:, :3, :3] = self.rotation
        self.transform_matrix[:, :3, 3] = self.translation
    def forward(self, right_se3, name):
        if name == "inverse":
            rotation = self.rotation.transpose(1, 2)
            # print("rotation-----", rotation.shape)
            # print("-self.translation-----", (-se3_dict["translation"]).shape)
            _translation = -self.translation.unsqueeze(2)
            translation = torch.bmm(rotation, _translation)
            # print("translation-----", translation.shape)
            return SE3(rotation=rotation, translation=translation.squeeze(2))
        elif name == "right_multiply_with_se3":
            chained_transform_matrix = torch.bmm(self.transform_matrix, right_se3.transform_matrix)
            chained_se3 = SE3(
                rotation=chained_transform_matrix[:, :3, :3],
                translation=chained_transform_matrix[:, :3, 3],
            )
            return chained_se3
    def inverse(self) -> "SE3":

        """Return the inverse of the current SE3 transformation.

        For example, if the current object represents target_SE3_src, we will return instead src_SE3_target.

        Returns:
            src_SE3_target: instance of SE3 class, representing
                inverse of SE3 transformation target_SE3_src
        """
        # rotation [3,3]
        # tensor [8, 3. 3]
        rotation = self.rotation.transpose(1, 2)
        print("rotation-----",rotation.shape)
        print("-self.translation-----", (-self.translation).shape)
        _translation = -self.translation.unsqueeze(2)
        translation = torch.bmm(rotation, _translation)
        print("translation-----", translation.shape)
        return SE3(rotation=rotation, translation=translation.squeeze(2))

    def compose(self, se3_dict, right_se3: "SE3") -> "SE3":
        """Compose (right multiply) this class' transformation matrix T with another SE3 instance.

        Algebraic representation: chained_se3 = T * right_se3

        Args:
            right_se3: another instance of SE3 class

        Returns:
            chained_se3: new instance of SE3 class
        """
        # print("self.transform_matrix-----", self.transform_matrix.shape)
        # print("right_se3.transform_matrix-----", right_se3.transform_matrix.shape)
        chained_transform_matrix = torch.bmm(se3_dict["transform_matrix"], right_se3["transform_matrix"])
        chained_se3 = SE3(
            rotation=chained_transform_matrix[:, :3, :3],
            translation=chained_transform_matrix[:, :3, 3],
        )
        return chained_se3

    def right_multiply_with_se3(self, se3_dict, right_se3: "SE3") -> "SE3":
        """Compose (right multiply) this class' transformation matrix T with another SE3 instance.

        Algebraic representation: chained_se3 = T * right_se3

        Args:
            right_se3: another instance of SE3 class

        Returns:
            chained_se3: new instance of SE3 class
        """
        return self.compose(se3_dict, right_se3)