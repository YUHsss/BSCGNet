import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .vgg import VGG

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

# for conv5
class ACCoM5(nn.Module):
    def __init__(self, cur_channel):
        super(ACCoM5, self).__init__()
        self.relu = nn.ReLU(True)

        # current conv
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)

        self.cur_all = BasicConv2d(4*cur_channel, cur_channel, 3, padding=1)
        self.cur_all_ca = ChannelAttention(cur_channel)
        self.cur_all_sa = SpatialAttention()

        # previous conv
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.pre_sa = SpatialAttention()

    def forward(self, x_pre, x_cur):
        # current conv
        x_cur_1 = self.cur_b1(x_cur)
        x_cur_2 = self.cur_b2(x_cur)
        x_cur_3 = self.cur_b3(x_cur)
        x_cur_4 = self.cur_b4(x_cur)
        x_cur_all = self.cur_all(torch.cat((x_cur_1, x_cur_2, x_cur_3, x_cur_4), 1))
        cur_all_ca = x_cur_all.mul(self.cur_all_ca(x_cur_all))
        cur_all_sa = x_cur_all.mul(self.cur_all_sa(cur_all_ca))

        # previois conv
        x_pre = self.downsample2(x_pre)
        pre_sa = x_cur_all.mul(self.pre_sa(x_pre))

        x_LocAndGlo = cur_all_sa + pre_sa + x_cur

        return x_LocAndGlo

# for conv1
class ACCoM1(nn.Module):
    def __init__(self, cur_channel):
        super(ACCoM1, self).__init__()
        self.relu = nn.ReLU(True)

        # current conv
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)

        self.cur_all = BasicConv2d(4*cur_channel, cur_channel, 3, padding=1)
        self.cur_all_ca = ChannelAttention(cur_channel)
        self.cur_all_sa = SpatialAttention()

        # latter conv
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lat_sa = SpatialAttention()

    def forward(self, x_cur, x_lat):
        # current conv
        x_cur_1 = self.cur_b1(x_cur)
        x_cur_2 = self.cur_b2(x_cur)
        x_cur_3 = self.cur_b3(x_cur)
        x_cur_4 = self.cur_b4(x_cur)
        x_cur_all = self.cur_all(torch.cat((x_cur_1, x_cur_2, x_cur_3, x_cur_4), 1))
        cur_all_ca = x_cur_all.mul(self.cur_all_ca(x_cur_all))
        cur_all_sa = x_cur_all.mul(self.cur_all_sa(cur_all_ca))

        # latter conv
        x_lat = self.upsample2(x_lat)
        lat_sa = x_cur_all.mul(self.lat_sa(x_lat))

        x_LocAndGlo = cur_all_sa + lat_sa + x_cur

        return x_LocAndGlo

    # for conv2/3/4
class ACCoM(nn.Module):
    def __init__(self, cur_channel):
        super(ACCoM, self).__init__()
        self.relu = nn.ReLU(True)

        # current conv
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)

        self.cur_all = BasicConv2d(4 * cur_channel, cur_channel, 3, padding=1)
        self.cur_all_ca = ChannelAttention(cur_channel)
        self.cur_all_sa = SpatialAttention()

        # previous conv
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.pre_sa = SpatialAttention()

        # latter conv
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lat_sa = SpatialAttention()

    def forward(self, x_pre, x_cur, x_lat):
        # current conv
        x_cur_1 = self.cur_b1(x_cur)
        x_cur_2 = self.cur_b2(x_cur)
        x_cur_3 = self.cur_b3(x_cur)
        x_cur_4 = self.cur_b4(x_cur)
        x_cur_all = self.cur_all(torch.cat((x_cur_1, x_cur_2, x_cur_3, x_cur_4), 1))
        cur_all_ca = x_cur_all.mul(self.cur_all_ca(x_cur_all))
        cur_all_sa = x_cur_all.mul(self.cur_all_sa(cur_all_ca))

        # previois conv
        x_pre = self.downsample2(x_pre)
        pre_sa = x_cur_all.mul(self.pre_sa(x_pre))

        # latter conv
        x_lat = self.upsample2(x_lat)
        lat_sa = x_cur_all.mul(self.lat_sa(x_lat))

        x_LocAndGlo = cur_all_sa + pre_sa + lat_sa + x_cur

        return x_LocAndGlo


class BAB_Decoder(nn.Module):
    def __init__(self, channel_1=1024, channel_2=512, channel_3=256, dilation_1=3, dilation_2=2):
        super(BAB_Decoder, self).__init__()

        self.conv1 = BasicConv2d(channel_1, channel_2, 3, padding=1)
        self.conv1_Dila = BasicConv2d(channel_2, channel_2, 3, padding=dilation_1, dilation=dilation_1)
        self.conv2 = BasicConv2d(channel_2, channel_2, 3, padding=1)
        self.conv2_Dila = BasicConv2d(channel_2, channel_2, 3, padding=dilation_2, dilation=dilation_2)
        self.conv3 = BasicConv2d(channel_2, channel_2, 3, padding=1)
        self.conv_fuse = BasicConv2d(channel_2*3, channel_3, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_dila = self.conv1_Dila(x1)

        x2 = self.conv2(x1)
        x2_dila = self.conv2_Dila(x2)

        x3 = self.conv3(x2)

        x_fuse = self.conv_fuse(torch.cat((x1_dila, x2_dila, x3), 1))

        return x_fuse


class decoder(nn.Module):
    def __init__(self, channel=512):
        super(decoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder5 = nn.Sequential(
            BAB_Decoder(512, 512, 512, 3, 2),
            nn.Dropout(0.5),
            TransBasicConv2d(512, 512, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
        )
        self.S5 = nn.Conv2d(512, 1, 3, stride=1, padding=1)

        self.decoder4 = nn.Sequential(
            BAB_Decoder(1024, 512, 256, 3, 2),
            nn.Dropout(0.5),
            TransBasicConv2d(256, 256, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S4 = nn.Conv2d(256, 1, 3, stride=1, padding=1)

        self.decoder3 = nn.Sequential(
            BAB_Decoder(512, 256, 128, 5, 3),
            nn.Dropout(0.5),
            TransBasicConv2d(128, 128, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S3 = nn.Conv2d(128, 1, 3, stride=1, padding=1)

        self.decoder2 = nn.Sequential(
            BAB_Decoder(256, 128, 64, 5, 3),
            nn.Dropout(0.5),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S2 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.decoder1 = nn.Sequential(
            BAB_Decoder(128, 64, 32, 5, 3)
        )
        self.S1 = nn.Conv2d(32, 1, 3, stride=1, padding=1)


    def forward(self, x5, x4, x3, x2, x1):
        # x5: 1/16, 512; x4: 1/8, 512; x3: 1/4, 256; x2: 1/2, 128; x1: 1/1, 64
        x5_up = self.decoder5(x5)
        s5 = self.S5(x5_up)

        x4_up = self.decoder4(torch.cat((x4, x5_up), 1))
        s4 = self.S4(x4_up)

        x3_up = self.decoder3(torch.cat((x3, x4_up), 1))
        s3 = self.S3(x3_up)

        x2_up = self.decoder2(torch.cat((x2, x3_up), 1))
        s2 = self.S2(x2_up)

        x1_up = self.decoder1(torch.cat((x1, x2_up), 1))
        s1 = self.S1(x1_up)

        return s1, s2, s3, s4, s5


class ACCoNet_VGG(nn.Module):
    def __init__(self, channel=32):
        super(ACCoNet_VGG, self).__init__()
        #Backbone model
        self.vgg = VGG('rgb')

        self.ACCoM5 = ACCoM5(512)
        self.ACCoM4 = ACCoM(512)
        self.ACCoM3 = ACCoM(256)
        self.ACCoM2 = ACCoM(128)
        self.ACCoM1 = ACCoM1(64)

        # self.agg2_rgbd = aggregation(channel)
        self.decoder_rgb = decoder(512)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x_rgb):
        x1_rgb = self.vgg.conv1(x_rgb)
        x2_rgb = self.vgg.conv2(x1_rgb)
        x3_rgb = self.vgg.conv3(x2_rgb)
        x4_rgb = self.vgg.conv4(x3_rgb)
        x5_rgb = self.vgg.conv5(x4_rgb)

        # up means update
        x5_ACCoM = self.ACCoM5(x4_rgb, x5_rgb)
        x4_ACCoM = self.ACCoM4(x3_rgb, x4_rgb, x5_rgb)
        x3_ACCoM = self.ACCoM3(x2_rgb, x3_rgb, x4_rgb)
        x2_ACCoM = self.ACCoM2(x1_rgb, x2_rgb, x3_rgb)
        x1_ACCoM = self.ACCoM1(x1_rgb, x2_rgb)

        s1, s2, s3, s4, s5 = self.decoder_rgb(x5_ACCoM, x4_ACCoM, x3_ACCoM, x2_ACCoM, x1_ACCoM)

        s3 = self.upsample2(s3)
        s4 = self.upsample4(s4)
        s5 = self.upsample8(s5)

        return s1, s2, s3, s4, s5, self.sigmoid(s1), self.sigmoid(s2), self.sigmoid(s3), self.sigmoid(s4), self.sigmoid(s5)

if __name__ == '__main__':
    # ras = Net().cuda()
    # input_tensor = torch.randn(1, 3, 256, 256).cuda()
    # x1,x2,x3,x4,x5 = ras(input_tensor)
    # print(x1.shape)
    # print(x2.shape)
    # print(x3.shape)
    # # print(x[5].shape)
    # print(x4.shape)
    # print(x5.shape)
    from thop import profile

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    model = ACCoNet_VGG()
    input = torch.randn(1, 3, 256, 256)
    output = model(input)
    flops, params = profile(model, inputs=(input,))
    print('flops(G): %.3f' % (flops / 1e+9))
    print('params(M): %.3f' % (params / 1e+6))

    # def opCounter(model):
    #     type_size = 4  # float
    #     params = list(model.parameters())
    #     k = 0
    #     for i in params:
    #         l = 1
    #         print("该层的结构：" + str(list(i.size())))
    #         for j in i.size():
    #             l *= j
    #         print("该层参数和：" + str(l))
    #         k = k + l
    #     print("总参数数量和：" + str(k))
    #     print('Model {} : params: {:4f}M'.format(model._get_name(), k * type_size / 1000 / 1000))
    #
    #
    # model = Net()
    # opCounter(model)
