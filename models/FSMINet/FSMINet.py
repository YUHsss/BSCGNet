import torch
import torch.nn as nn
import torchvision

from basic import *

def change(x):
    x1 = torch.where(x == 1, torch.full_like(x, 1-(1e-5)), x)
    x2 = torch.where(x1 == 0, torch.full_like(x1, (1e-5)), x1)

    return x2

class FSMINet(nn.Module):
    def __init__(self):
        super(FSMINet,self).__init__()

        ## -------------Encoder--------------
        #stage1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,bias=False) # 384 64
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #stage 2
        self.encoder2 = FSM(64, 128, Squeeze = False) #192 128
        #stage 3
        self.encoder3 = FSM(128, 256) #96 256
        #stage 4
        self.encoder4 = FSM(256, 512) #48 512
        #stage 5d
        self.encoder5 = FSM(512, 1024) #24 1024

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ## -------------Decoder--------------
        #stage 5d
        self.decoder5 = FSM(1024, 512)

        #stage 4d
        self.decoder4 = FSM(512, 256)

        #stage 3d
        self.decoder3 = FSM(256, 128)

        #stage 2d
        self.decoder2 = FSM(128, 64)

        #stage 1d
        self.decoder1 = FSM(64, 64,Squeeze=False)

        ## -------------Bilinear Upsampling--------------
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True)
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear',align_corners=True)
        self.upscore16 = nn.Upsample(scale_factor=16, mode='bilinear',align_corners=True)

        ## -------------Output--------------
        self.conv_out0 = nn.Conv2d(5,1,1,padding=0)
        self.conv_out1 = nn.Conv2d(64,1,1,padding=0)
        self.conv_out2 = nn.Conv2d(64,1,1,padding=0)
        self.conv_out3 = nn.Conv2d(128,1,1,padding=0)
        self.conv_out4 = nn.Conv2d(256,1,1,padding=0)
        self.conv_out5 = nn.Conv2d(512,1,1,padding=0)

    def forward(self,x):

        ## -------------Encoder-------------
        score = x
        score1 = self.relu(self.bn1(self.conv1(score)))

        score2 = self.maxpool(score1)
        score2 = self.encoder2(score2)

        score3 = self.maxpool(score2)
        score3 = self.encoder3(score3)

        score4 = self.maxpool(score3)
        score4 = self.encoder4(score4)

        score5 = self.maxpool(score4)
        score5 = self.encoder5(score5)

        ## -------------Decoder--------------
        #stage 5d
        scored5 = self.decoder5(score5)
        t = self.upscore2(scored5)
        #stage 4d
        scored4 = self.decoder4(score4 + t)
        t = self.upscore2(scored4)
        #stage 3d
        scored3 = self.decoder3(score3 + t)
        t = self.upscore2(scored3)
        #stage 2d
        scored2 = self.decoder2(score2 + t)
        t = self.upscore2(scored2)
        #stage 1d
        scored1 = self.decoder1(score1 + t)

        ## -------------Output--------------
        out1 = self.conv_out1(scored1)

        out2 = self.conv_out2(scored2)
        out2 = self.upscore2(out2)

        out3 = self.conv_out3(scored3)
        out3 = self.upscore4(out3)

        out4 = self.conv_out4(scored4)
        out4 = self.upscore8(out4)

        out5 = self.conv_out5(scored5)
        out5 = self.upscore16(out5)

        out0 = torch.cat((out1, out2, out3, out4, out5),1)
        out0 = self.conv_out0(out0)

        return change(torch.sigmoid(out0)), change(torch.sigmoid(out1)), change(torch.sigmoid(out2)), \
            change(torch.sigmoid(out3)), change(torch.sigmoid(out4)), change(torch.sigmoid(out5))

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
    from tools.get_model_size import *

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    model = FSMINet()
    input = torch.randn(1, 3, 256, 256)
    output = model(input)
    flops, params = profile(model, inputs=(input,))
    print('flops(G): %.3f' % (flops / 1e+9))
    print('params(M): %.3f' % (params / 1e+6))
    getModelSize(model)
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
