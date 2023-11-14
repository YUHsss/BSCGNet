import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from .vgg_base import VGG


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, reduction=16):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)

        if residual.shape[1] != x.shape[1]:
            residual = self.channel_conv(residual)
        x = x + residual
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #  Backbone model VGG
        self.vgg = VGG()

        #  Backbone model Resnet
        # resnet = torchvision.models.resnet34(True)
        # self.preconv = nn.Conv2d(3, 64, 3, padding=1)
        # self.prebn = nn.BatchNorm2d(64)
        # self.prerelu = nn.ReLU(inplace=True)
        # #
        # self.encoder1 = resnet.layer1
        # self.encoder2 = resnet.layer2
        # self.encoder3 = resnet.layer3
        # self.encoder4 = resnet.layer4
        # self.maxpool = nn.MaxPool2d(2, 2, ceil_mode=True)
        # self.encoder5_1 = BasicBlock(512, 512)
        # self.encoder5_2 = BasicBlock(512, 512)
        # self.encoder5_3 = BasicBlock(512, 512)

        self.d0 = nn.Conv2d(64, 64, kernel_size=1)
        self.d1 = nn.Conv2d(128, 64, kernel_size=1)
        self.d2 = nn.Conv2d(256, 64, kernel_size=1)
        self.d3 = nn.Conv2d(512, 64, kernel_size=1)
        self.res1 = BasicBlock(64, 64)
        self.res2 = BasicBlock(64, 64)
        self.res3 = BasicBlock(64, 64)
        self.SE128 = SEAttention(channel=128)
        self.SE512 = SEAttention(channel=512)
        self.SE256 = SEAttention(channel=256)
        self.SE64 = SEAttention(channel=64)

        self.flow_make = nn.Conv2d(128, 2, kernel_size=3, padding=1, bias=False)
        self.up1 = up(512 + 512, 512)
        self.up2 = up(512 + 256, 256)
        self.up3 = up(256 + 128, 128)
        self.up4 = up(128 + 64, 64)
        self.out = nn.Conv2d(64 * 3, 1, kernel_size=1, bias=False)
        self.out1 = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.out2 = nn.Conv2d(512, 1, kernel_size=1, bias=False)
        self.out3 = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        self.out4 = nn.Conv2d(128, 1, kernel_size=1, bias=False)
        self.out5 = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.d_x5 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(True))
        self.edg_at = nn.Sequential(nn.Conv2d(64 * 2, 1, kernel_size=1, bias=False),
                                    nn.Sigmoid())
        self.edg_loss = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.conv1x1_fk1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(True))
        self.conv1x1_fk2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(True))
        self.conv1x1_fk3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(True))
        self.conv3x3_fk1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(64),
                                         nn.Sigmoid())
        self.conv3x3_fk1_r = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                           nn.BatchNorm2d(64),
                                           nn.ReLU(True))
        self.conv3x3_fk2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(64),
                                         nn.Sigmoid())
        self.conv3x3_fk2_r = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                           nn.BatchNorm2d(64),
                                           nn.ReLU(True))
        self.conv3x3_fk3_r = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                           nn.BatchNorm2d(64),
                                           nn.ReLU(True))
        self.conv3x3_fk4_r = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                           nn.BatchNorm2d(64),
                                           nn.ReLU(True))
        self.conv3x3_fk3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(True))
        self.conv3x3_fk4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(True))
        self.conv3x3_fk5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(True))
        self.conv3x3_fk6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(True))
        self.conv1x1_fk12 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=1),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(True))
        self.conv1x1_fk34 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=1),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(True))

        self.dc1_conv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(True))

        self.dc2_conv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, dilation=2, padding=2),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(True))
        self.dc3_conv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, dilation=3, padding=3),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(True))
        self.dc4_conv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, dilation=4, padding=4),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(True))
        self.dc5_conv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(True))

        self.dc6_conv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, dilation=2, padding=2),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(True))
        self.dc7_conv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, dilation=3, padding=3),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(True))
        self.dc8_conv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, dilation=4, padding=4),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(True))
        self.dc_fuse_conv = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=1),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(True))

        self.dc_fuse_conv1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                           nn.BatchNorm2d(64),
                                           nn.ReLU(True))
        self.dc_fuse_conv2 = nn.Sequential(nn.Conv2d(64 * 4, 64, kernel_size=3, padding=1),
                                           nn.BatchNorm2d(64),
                                           nn.ReLU(True))
        self.dc_fuse_conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                           nn.BatchNorm2d(64),
                                           nn.ReLU(True))
        # self.d0 = nn.Conv2d(512, 256, kernel_size=1)
        # self.d1 = nn.Conv2d(256, 128, kernel_size=1)
        # self.d2 = nn.Conv2d(128, 64, kernel_size=1)
        # self.no_bpro = nn.Sequential(nn.Conv2d(64*4,64,1),
        #                              nn.BatchNorm2d(64),
        #                              nn.ReLU(True))
    def forward(self, x):
        x_size = x.size()
        # TODO VGG16
        x1 = self.vgg.conv1(x)
        x2 = self.vgg.conv2(x1)
        x3 = self.vgg.conv3(x2)
        x4 = self.vgg.conv4(x3)
        x5 = self.vgg.conv5(x4)

        # TODO Resnet 34
        # x1 = self.encoder1(self.prerelu(self.prebn(self.preconv(x))))
        # x2 = self.encoder2(x1)
        # x3 = self.encoder3(x2)
        # x4 = self.encoder4(x3)
        #
        # score1_5 = self.maxpool(x4)
        # score1_5 = self.encoder5_1(score1_5)
        # score1_5 = self.encoder5_2(score1_5)
        # x5 = self.encoder5_3(score1_5)

        # TODO:边界保护优化(这个模块消融实验就单纯把4层信息加起来，不做优化处理)
        edg1 = F.interpolate(self.d0(x1), x_size[2:],
                             mode='bilinear', align_corners=True)

        edg1 = self.res1(edg1)
        # sss = edg1.clone()
        edg2 = F.interpolate(self.d1(x2), x_size[2:],
                             mode='bilinear', align_corners=True)

        edg12 = self.flow_make(self.SE128(torch.cat([edg1, edg2], 1)))

        norm = torch.tensor([[[x_size[2:]]]]).type_as(edg2).to(edg2.device) \
            # generate grid
        w = torch.linspace(-1.0, 1.0, x_size[3]).view(-1, 1).repeat(1, x_size[3])
        h = torch.linspace(-1.0, 1.0, x_size[2]).repeat(x_size[2], 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(x_size[0], 1, 1, 1).type_as(edg2).to(edg2.device)
        grid = grid + edg12.permute(0, 2, 3, 1) / norm
        output1 = F.grid_sample(edg2, grid) + edg1

        output1_short = output1.clone()
        output1 = self.res2(output1)
        edg3 = F.interpolate(self.d2(x3), x_size[2:],
                             mode='bilinear', align_corners=True)
        edg23 = self.flow_make(self.SE128(torch.cat([output1, edg3], 1)))
        norm = torch.tensor([[[x_size[2:]]]]).type_as(edg3).to(edg3.device) \
            # generate grid
        w = torch.linspace(-1.0, 1.0, x_size[3]).view(-1, 1).repeat(1, x_size[3])
        h = torch.linspace(-1.0, 1.0, x_size[2]).repeat(x_size[2], 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(x_size[0], 1, 1, 1).type_as(edg3).to(edg3.device)
        grid = grid + edg23.permute(0, 2, 3, 1) / norm
        output2 = F.grid_sample(edg3, grid) + output1_short
        output2_short = output2.clone()

        output2 = self.res3(output2)
        edg4 = F.interpolate(self.d3(x4), x_size[2:],
                             mode='bilinear', align_corners=True)
        edg34 = self.flow_make(self.SE128(torch.cat([output2, edg4], 1)))
        norm = torch.tensor([[[x_size[2:]]]]).type_as(edg4).to(edg4.device) \
            # generate grid
        w = torch.linspace(-1.0, 1.0, x_size[3]).view(-1, 1).repeat(1, x_size[3])
        h = torch.linspace(-1.0, 1.0, x_size[2]).repeat(x_size[2], 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(x_size[0], 1, 1, 1).type_as(edg4).to(edg4.device)
        grid = grid + edg34.permute(0, 2, 3, 1) / norm
        output3 = F.grid_sample(edg4, grid) + output2_short


        edg_ls = self.edg_loss(output3)

        # TODO:边界反馈（这个的消融实验就是不做反馈机制，直接将编码解码cat）
        edg_att = torch.cat([output3, F.interpolate(self.d_x5(x5),
                                                    size=x_size[2:], mode='bilinear')], 1)
        edg_att = self.edg_at(edg_att)

        edg_gd1 = (F.interpolate(edg_att, size=x5.size()[2:], mode='bilinear') + 1) * x5
        edg_gd1 = self.SE512(edg_gd1)  # 512 16 16

        edg_gd2 = (F.interpolate(edg_att, size=x4.size()[2:], mode='bilinear') + 1) * x4
        edg_gd2 = self.SE512(edg_gd2)  # 512 32 32

        edg_gd3 = (F.interpolate(edg_att, size=x3.size()[2:], mode='bilinear') + 1) * x3
        edg_gd3 = self.SE256(edg_gd3)  # 256 64 64

        edg_gd4 = (F.interpolate(edg_att, size=x2.size()[2:], mode='bilinear') + 1) * x2
        edg_gd4 = self.SE128(edg_gd4)  # 128 128 128

        edg_gd5 = (F.interpolate(edg_att, size=x1.size()[2:], mode='bilinear') + 1) * x1

        edg_gd5 = self.SE64(edg_gd5)  # 64 256 256

        # TODO: 在保留边界信息上解码 + 解码用残差（如果效果不好考虑直接cat编码来解码）
        decoder1 = self.up1(edg_gd1, edg_gd2)  # 512 32 32
        decoder2 = self.up2(decoder1, edg_gd3)  # 256 64 64
        decoder3 = self.up3(decoder2, edg_gd4)  # 128 128 128
        decoder4 = self.up4(decoder3, edg_gd5)  # 64 256 256
        # decoder1 = self.up1(x5, x4)  # 512 32 32
        # decoder2 = self.up2(decoder1, x3)  # 256 64 64
        # decoder3 = self.up3(decoder2, x2)  # 128 128 128
        # decoder4 = self.up4(decoder3, x1)  # 64 256 256

        # baseline
        # decoder1 = F.interpolate(x5, scale_factor=2, mode="bilinear") + x4  # 512 32 32
        # decoder2 = F.interpolate(self.d0(decoder1), scale_factor=2, mode="bilinear") + x3  # 256 64 64
        # decoder3 = F.interpolate(self.d1(decoder2), scale_factor=2, mode="bilinear") + x2  # 128 128 128
        # decoder4 = F.interpolate(self.d2(decoder3), scale_factor=2, mode="bilinear") + x1  # 64 256 256

        # TODO: try一try语义反馈(消融实验直接就输出深监督解码4层)
        fk1 = F.interpolate(decoder4, size=decoder1.size()[2:], mode='bilinear')
        fk2 = F.interpolate(decoder4, size=decoder2.size()[2:], mode='bilinear')
        fk3 = F.interpolate(decoder4, size=decoder3.size()[2:], mode='bilinear')

        decoder1_fk = self.conv1x1_fk1(decoder1)
        decoder2_fk = self.conv1x1_fk2(decoder2)
        decoder3_fk = self.conv1x1_fk3(decoder3)

        # 第一次反馈
        fk1_dec1 = F.interpolate(fk1 + decoder1_fk, scale_factor=2,
                                 mode='bilinear')  # 64 64 64
        fk1_dec1 = self.conv3x3_fk1_r(fk1_dec1)
        fk1_dec1_att = self.conv3x3_fk1(fk1_dec1)
        fk2_dec2 = fk2 + decoder2_fk  # 64 64 64
        fk2_dec2 = self.conv3x3_fk2_r(fk2_dec2)
        fk2_dec2_att = self.conv3x3_fk1(fk2_dec2)

        # cross
        fk1_dec1_cross = self.conv3x3_fk3((fk1_dec1_att + 1) * fk2_dec2)
        fk2_dec2_cross = self.conv3x3_fk4((fk2_dec2_att + 1) * fk1_dec1)
        fk12_fuse = self.conv1x1_fk12(torch.cat([fk1_dec1_cross, fk2_dec2_cross], 1))
        dc1 = self.dc1_conv(fk12_fuse)
        dc2 = self.dc2_conv(fk12_fuse + dc1)
        dc3 = self.dc3_conv(fk12_fuse + dc2)
        dc4 = self.dc4_conv(fk12_fuse + dc3)
        dc_fuse = self.dc_fuse_conv(torch.cat([dc1, dc2, dc3, dc4], 1)) + fk12_fuse
        dc_fuse = self.dc_fuse_conv1(dc_fuse)  # 64 64 64

        #
        # # 第二次反馈

        dc_fuse_up = F.interpolate(dc_fuse, scale_factor=2, mode='bilinear')  # 64 128 128

        fk3_dec3 = fk3 + decoder3_fk  # 64 128 128

        fk3_dec3 = self.conv3x3_fk3_r(fk3_dec3)
        fk3_dec3_att = self.conv3x3_fk1(fk3_dec3)

        dc_fuse_up = self.conv3x3_fk4_r(dc_fuse_up)
        dc_fuse_up_att = self.conv3x3_fk1(dc_fuse_up)

        fk3_dec3_cross = self.conv3x3_fk5((fk3_dec3_att + 1) * dc_fuse_up)
        dc_fuse_up_cross = self.conv3x3_fk6((dc_fuse_up_att + 1) * fk3_dec3)
        fk34_fuse = self.conv1x1_fk34(torch.cat([fk3_dec3_cross, dc_fuse_up_cross], 1))  # 64 128 128
        dc5 = self.dc5_conv(fk34_fuse)
        dc6 = self.dc6_conv(fk34_fuse + dc5)
        dc7 = self.dc7_conv(fk34_fuse + dc6)
        dc8 = self.dc8_conv(fk34_fuse + dc7)
        dc_fuse1 = self.dc_fuse_conv2(torch.cat([dc5, dc6, dc7, dc8], 1)) + fk34_fuse
        dc_fuse1 = self.dc_fuse_conv3(dc_fuse1)  # 64 128 128

        dc_fuse1_up = F.interpolate(dc_fuse1, scale_factor=2, mode='bilinear')
        refine = torch.cat([decoder4, output3, dc_fuse1_up], 1)

        refine = self.out(refine)

        #  TODO: 深监督

        fk = self.out1(dc_fuse1_up)

        d1_out = F.interpolate(self.out2(decoder1), size=[256, 256], mode='bilinear')
        d2_out = F.interpolate(self.out3(decoder2), size=[256, 256], mode='bilinear')
        d3_out = F.interpolate(self.out4(decoder3), size=[256, 256], mode='bilinear')
        d4_out = F.interpolate(self.out5(decoder4), size=[256, 256], mode='bilinear')

        return refine, fk, d4_out, d3_out, d2_out, d1_out

if __name__ == '__main__':
    from thop import profile
    from tools.get_model_size import *

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    model = Net()
    input = torch.randn(1, 3, 256, 256)
    output = model(input)
    flops, params = profile(model, inputs=(input,))
    print('flops(G): %.3f' % (flops / 1e+9))
    print('params(M): %.3f' % (params / 1e+6))
    getModelSize(model)
