import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
import matplotlib.pyplot as PLT
import matplotlib.cm as mpl_color_map
from .layout_model import Conv3x3

def feature_selection(input, dim, index):
    # feature selection
    # input: [N, ?, ?, ...]
    # dim: scalar > 0
    # index: [N, idx]
    views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]
    expanse = list(input.size())
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


class CrossViewTransformer(nn.Module):
    def __init__(self, in_dim):
        super(CrossViewTransformer, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.f_conv = nn.Conv2d(in_channels=in_dim * 2, out_channels=in_dim, kernel_size=3, stride=1, padding=1,
                                bias=True)

        self.res_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.query_conv_depth = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv_depth = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv_depth = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv1 = Conv3x3(512, 128)
        self.conv2 = Conv3x3(128, 128)
        self.pool = nn.MaxPool2d(2)

    def forward(self, front_x, cross_x, front_x_hat, depth_feature):
        depth_feature = self.pool(self.conv1(depth_feature))
        depth_feature = self.pool(self.conv2(depth_feature))
        # print("depth_feature shape******************", depth_feature.shape)
        # print("front_x shape******************", front_x.shape)
        # print("cross_x shape******************", cross_x.shape)
        # print("front_x_hat shape******************", front_x_hat.shape)
        m_batchsize, C, width, height = front_x.size()
        proj_query = self.query_conv(cross_x).view(m_batchsize, -1, width * height)  # B x C x (N)
        proj_key = self.key_conv(front_x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x C x (W*H)
        # print("proj_query shape******************", proj_query.shape)
        # print("proj_key shape******************", proj_key.shape)

        energy = torch.bmm(proj_key, proj_query)  # transpose check

        # print("energy shape******************", energy.shape)
        front_star, front_star_arg = torch.max(energy, dim=1)
        # print("front_star shape******************", front_star.shape)
        proj_value = self.value_conv(front_x_hat).view(m_batchsize, -1, width * height)  # B x C x N
        # print("proj_value shape******************", proj_value.shape)
        T = feature_selection(proj_value, 2, front_star_arg).view(front_star.size(0), -1, width, height)
        # print("T shape******************", T.shape)
        S = front_star.view(front_star.size(0), 1, width, height)
        # print("S shape******************", S.shape)
        front_res = torch.cat((front_x, T), dim=1)
        # print("S shape******************concat", front_res.shape)
        front_res = self.f_conv(front_res)
        # print("S shape******************f_conv", front_res.shape)
        front_res = front_res * S
        # print(front_res.shape)
        output = front_x + front_res
        # print("output shape******************", output.shape)
        proj_query_depth = self.query_conv_depth(cross_x).view(m_batchsize, -1, width * height)  # B x C x (N)
        proj_key_depth = self.key_conv_depth(front_x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x C x (W*H)
        proj_value_depth = self.value_conv_depth(depth_feature).view(m_batchsize, -1, width, height)
        # print("proj_query_depth shape******************", proj_query_depth.shape)
        # print("proj_key_depth shape******************", proj_key_depth.shape)
        # print("proj_value_depth shape******************", proj_value_depth.shape)
        attn =  proj_key_depth @ proj_query_depth
        # print("attn shape******************", attn.shape)
        attn,_ = torch.max(attn, dim=1)
        attn = attn.view(attn.size(0), 1, width, height)
        # print("attn shape******************", attn.shape)
        x_depth = attn @ proj_value_depth
        # x_depth = self.proj(x_depth)
        # print("x_depth shape******************", x_depth.shape)
        output = output + x_depth
        return output, S, attn


if __name__ == '__main__':
    # features = torch.arange(0, 24)
    features = torch.arange(0, 65536)
    features = torch.where(features < 20, features, torch.zeros_like(features))
    # features = features.view([2, 3, 4]).float()
    features = features.view([8, 128, 8, 8]).float()

    features2 = torch.arange(0, 65536)
    features2 = torch.where(features2 < 20, features2, torch.zeros_like(features2))
    # features = features.view([2, 3, 4]).float()
    features2 = features2.view([8, 128, 8, 8]).float()

    features3 = torch.arange(0, 65536)
    features3 = torch.where(features3 < 20, features3, torch.zeros_like(features3))
    # features = features.view([2, 3, 4]).float()
    features3 = features3.view([8, 128, 8, 8]).float()

    attention3 = CrossViewTransformer(128)
    print(attention3(features, features2, features3).shape)
