import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Conv1x1, Conv3x3, CRPBlock, upsample


class DepthDecoder(nn.Module):
    def __init__(self,  num_ch_enc):
        super(DepthDecoder, self).__init__()

        bottleneck = 256
        stage = 4
        self.do = nn.Dropout(p=0.5)

        self.reduce4 = Conv1x1(num_ch_enc[4], 512, bias=False)
        self.reduce3 = Conv1x1(num_ch_enc[3], bottleneck, bias=False)
        self.reduce2 = Conv1x1(num_ch_enc[2], bottleneck, bias=False)
        self.reduce1 = Conv1x1(num_ch_enc[1], bottleneck, bias=False)

        self.iconv4 = Conv3x3(512, bottleneck)
        self.iconv3 = Conv3x3(bottleneck*2+1, bottleneck)
        self.iconv2 = Conv3x3(bottleneck*2+1, bottleneck)
        self.iconv1 = Conv3x3(bottleneck*2+1, bottleneck)

        self.crp4 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp3 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp2 = self._make_crp(bottleneck, bottleneck, stage)
        self.crp1 = self._make_crp(bottleneck, bottleneck, stage)

        self.merge4 = Conv3x3(bottleneck, bottleneck)
        self.merge3 = Conv3x3(bottleneck, bottleneck)
        self.merge2 = Conv3x3(bottleneck, bottleneck)
        self.merge1 = Conv3x3(bottleneck, bottleneck)

        # disp
        self.disp4 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())
        self.disp3 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())
        self.disp2 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())
        self.disp1 = nn.Sequential(Conv3x3(bottleneck, 1), nn.Sigmoid())

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)

    def forward(self, input_features, frame_id=0):
        # print("depth encoder last shape", len(input_features))
        # for i in range(len(input_features)):
        #     print("depth encoder shape "+ str(i)+"----", input_features[i].shape)
        self.outputs = {}
        l0, l1, l2, l3, l4 = input_features

        l4 = self.do(l4)
        l3 = self.do(l3)
        # print("after dropout " + " l4", l4.shape)
        # print("after dropout " + " l3", l3.shape)
        x4 = self.reduce4(l4)
        # print("after reduce4 " + " x4", x4.shape)
        x4 = self.iconv4(x4)
        # print("after iconv4 " + " x4", x4.shape)
        x4 = F.leaky_relu(x4)
        # print("after leaky_relu " + " x4", x4.shape)
        x4 = self.crp4(x4)
        # print("after crp4 " + " x4", x4.shape)
        x4 = self.merge4(x4)
        # print("after merge4 " + " x4", x4.shape)
        x4 = F.leaky_relu(x4)
        # print("after leaky_relu " + " x4", x4.shape)
        x4 = upsample(x4)
        # print("after upsample " + " x4", x4.shape)
        disp4 = self.disp4(x4)
        # print("after disp4 " + " x4", disp4.shape)


        x3 = self.reduce3(l3)
        # print("after reduce3 " + " x3", x3.shape)
        x3 = torch.cat((x3, x4, disp4), 1)
        # print("after cat " + " x3", x3.shape)
        x3 = self.iconv3(x3)
        # print("after iconv3 " + " x3", x3.shape)
        x3 = F.leaky_relu(x3)
        # print("after leaky_relu " + " x3", x3.shape)
        x3 = self.crp3(x3)
        # print("after crp3 " + " x3", x3.shape)
        x3 = self.merge3(x3)
        # print("after merge3 " + " x3", x3.shape)
        x3 = F.leaky_relu(x3)
        # print("after leaky_relu " + " x3", x3.shape)
        x3 = upsample(x3)
        # print("after upsample " + " x3", x3.shape)
        disp3 = self.disp3(x3)
        # print("after disp3 " + " disp3", disp3.shape)


        x2 = self.reduce2(l2)
        # print("after reduce2 " + " x2", x2.shape)
        x2 = torch.cat((x2, x3 , disp3), 1)
        # print("after cat " + " x2", x2.shape)
        x2 = self.iconv2(x2)
        # print("after iconv2 " + " x2", x2.shape)
        x2 = F.leaky_relu(x2)
        # print("after leaky_relu " + " x2", x2.shape)
        x2 = self.crp2(x2)
        # print("after leaky_relu " + " x2", x2.shape)
        x2 = self.merge2(x2)
        # print("after merge2 " + " x2", x2.shape)
        x2 = F.leaky_relu(x2)
        # print("after leaky_relu " + " x2", x2.shape)
        x2 = upsample(x2)
        # print("after upsample " + " x2", x2.shape)
        disp2 = self.disp2(x2)
        # print("after disp2 " + " disp2", disp2.shape)

        x1 = self.reduce1(l1)
        # print("after reduce1 " + " x1", x1.shape)
        x1 = torch.cat((x1, x2, disp2), 1)
        # print("after cat " + " x1", x1.shape)
        x1 = self.iconv1(x1)
        # print("after iconv1 " + " x1", x1.shape)
        x1 = F.leaky_relu(x1)
        # print("after leaky_relu " + " x1", x1.shape)
        x1 = self.crp1(x1)
        # print("after crp1 " + " x1", x1.shape)
        x1 = self.merge1(x1)
        # print("after merge1 " + " x1", x1.shape)
        x1 = F.leaky_relu(x1)
        # print("after leaky_relu " + " x1", x1.shape)
        x1 = upsample(x1)
        # print("after upsample " + " x1", x1.shape)
        disp1 = self.disp1(x1)
        # print("after disp1 " + " disp1", disp1.shape)

        self.outputs[("disp", frame_id, 3)] = disp4
        self.outputs[("disp", frame_id, 2)] = disp3
        self.outputs[("disp", frame_id, 1)] = disp2
        self.outputs[("disp", frame_id, 0)] = disp1

        return self.outputs
