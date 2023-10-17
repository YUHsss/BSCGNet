import torch.nn as nn
import torch

class FSM(nn.Module):
    def __init__(self, inchannels, outchannels, Squeeze = True):
        super(FSM, self).__init__()

        if Squeeze == True:
            #part 1
            self.conv_C1 = nn.Conv2d(inchannels, outchannels // 4, 1, padding=0)
            self.relu_C1 = nn.ReLU(inplace=False)

            self.conv_C2 = nn.Conv2d(outchannels // 4, outchannels // 4, 3, padding=1, groups=4)
            self.BN_C2 = nn.BatchNorm2d(outchannels // 4)
            self.relu_C2 = nn.ReLU(inplace=False)

            self.conv_C3 = nn.Conv2d(outchannels // 4, outchannels // 4, 3, padding=1, groups=4)
            self.BN_C3 = nn.BatchNorm2d(outchannels // 4)
            self.relu_C3 = nn.ReLU(inplace=False)

            self.conv_C4 = nn.Conv2d(outchannels // 4, outchannels // 4, 3, padding=1, groups=4)
            self.BN_C4 = nn.BatchNorm2d(outchannels // 4)
            self.relu_C4 = nn.ReLU(inplace=False)

            self.conv_C5 = nn.Conv2d(outchannels // 4, (outchannels // 8) * 3, 3, padding=1, groups=4)
            self.BN_C5 = nn.BatchNorm2d((outchannels // 8) * 3)
            self.relu_C5 = nn.ReLU(inplace=False)
            #part 2
            self.conv_C6 = nn.Conv2d(inchannels, outchannels // 4, 1, padding=0)
            self.relu_C6 = nn.ReLU(inplace=False)

            self.conv_DiC1 = nn.Conv2d(outchannels // 4, outchannels // 4, 3, dilation=2, padding=2, groups=4)
            self.BN_DiC1 = nn.BatchNorm2d(outchannels // 4)
            self.relu_DiC1 = nn.ReLU(inplace=False)

            self.conv_DiC2 = nn.Conv2d(outchannels // 4, outchannels // 4, 3, dilation=2, padding=2, groups=4)
            self.BN_DiC2 = nn.BatchNorm2d(outchannels // 4)
            self.relu_DiC2 = nn.ReLU(inplace=False)

            self.conv_DiC3 = nn.Conv2d(outchannels // 4, outchannels // 4, 3, dilation=2, padding=2, groups=4)
            self.BN_DiC3 = nn.BatchNorm2d(outchannels // 4)
            self.relu_DiC3 = nn.ReLU(inplace=False)

            self.conv_C7 = nn.Conv2d(outchannels // 4, (outchannels // 8) * 3, 3, padding=1, groups=4)
            self.BN_C7 = nn.BatchNorm2d((outchannels // 8) * 3)
            self.relu_C7 = nn.ReLU(inplace=False)
            #part 3
            self.conv_C8 = nn.Conv2d(inchannels, outchannels // 8, 1, padding=0)
            self.relu_C8 = nn.ReLU(inplace=False)

            self.conv_DiC4 = nn.Conv2d(outchannels // 8, outchannels // 8, 3, dilation=6, padding=6, groups=4)
            self.BN_DiC4 = nn.BatchNorm2d(outchannels // 8)
            self.relu_DiC4 = nn.ReLU(inplace=False)

            self.conv_DiC5 = nn.Conv2d(outchannels // 8, outchannels // 8, 3, dilation=4, padding=4, groups=4)
            self.BN_DiC5 = nn.BatchNorm2d(outchannels // 8)
            self.relu_DiC5 = nn.ReLU(inplace=False)

            self.conv_DiC6 = nn.Conv2d(outchannels // 8, outchannels // 8, 3, dilation=2, padding=2, groups=4)
            self.BN_DiC6 = nn.BatchNorm2d(outchannels // 8)
            self.relu_DiC6 = nn.ReLU(inplace=False)

            self.conv_C9 = nn.Conv2d(outchannels // 8, (outchannels // 8) * 2, 3, padding=1, groups=4)
            self.BN_C9 = nn.BatchNorm2d((outchannels // 8) * 2)
            self.relu_C9 = nn.ReLU(inplace=False)

        else:
            #part 1
            self.conv_C1 = nn.Conv2d(inchannels, inchannels, 1, padding=0)
            self.relu_C1 = nn.ReLU(inplace=False)

            self.conv_C2 = nn.Conv2d(inchannels, inchannels, 3, padding=1, groups=4)
            self.BN_C2 = nn.BatchNorm2d(inchannels)
            self.relu_C2 = nn.ReLU(inplace=False)

            self.conv_C3 = nn.Conv2d(inchannels, inchannels, 3, padding=1, groups=4)
            self.BN_C3 = nn.BatchNorm2d(inchannels)
            self.relu_C3 = nn.ReLU(inplace=False)

            self.conv_C4 = nn.Conv2d(inchannels, inchannels, 3, padding=1, groups=4)
            self.BN_C4 = nn.BatchNorm2d(inchannels)
            self.relu_C4 = nn.ReLU(inplace=False)

            self.conv_C5 = nn.Conv2d(inchannels, (outchannels // 8) * 3, 3, padding=1, groups=4)
            self.BN_C5 = nn.BatchNorm2d((outchannels // 8) * 3)
            self.relu_C5 = nn.ReLU(inplace=False)
            #part 2
            self.conv_C6 = nn.Conv2d(inchannels, inchannels, 1, padding=0)
            self.relu_C6 = nn.ReLU(inplace=False)

            self.conv_DiC1 = nn.Conv2d(inchannels, inchannels, 3, dilation=2, padding=2, groups=4)
            self.BN_DiC1 = nn.BatchNorm2d(inchannels)
            self.relu_DiC1 = nn.ReLU(inplace=False)

            self.conv_DiC2 = nn.Conv2d(inchannels, inchannels, 3, dilation=2, padding=2, groups=4)
            self.BN_DiC2 = nn.BatchNorm2d(inchannels)
            self.relu_DiC2 = nn.ReLU(inplace=False)

            self.conv_DiC3 = nn.Conv2d(inchannels, inchannels, 3, dilation=2, padding=2, groups=4)
            self.BN_DiC3 = nn.BatchNorm2d(inchannels)
            self.relu_DiC3 = nn.ReLU(inplace=False)

            self.conv_C7 = nn.Conv2d(inchannels, (outchannels // 8) * 3, 3, padding=1, groups=4)
            self.BN_C7 = nn.BatchNorm2d((outchannels // 8) * 3)
            self.relu_C7 = nn.ReLU(inplace=False)
            #part 3
            self.conv_C8 = nn.Conv2d(inchannels, inchannels // 2, 1, padding=0)
            self.relu_C8 = nn.ReLU(inplace=False)

            self.conv_DiC4 = nn.Conv2d(inchannels // 2, inchannels // 2, 3, dilation=6, padding=6, groups=4)
            self.BN_DiC4 = nn.BatchNorm2d(inchannels // 2)
            self.relu_DiC4 = nn.ReLU(inplace=False)

            self.conv_DiC5 = nn.Conv2d(inchannels // 2, inchannels // 2, 3, dilation=4, padding=4, groups=4)
            self.BN_DiC5 = nn.BatchNorm2d(inchannels // 2)
            self.relu_DiC5 = nn.ReLU(inplace=False)

            self.conv_DiC6 = nn.Conv2d(inchannels // 2, inchannels // 2, 3, dilation=2, padding=2, groups=4)
            self.BN_DiC6 = nn.BatchNorm2d(inchannels // 2)
            self.relu_DiC6 = nn.ReLU(inplace=False)

            self.conv_C9 = nn.Conv2d(inchannels // 2, (outchannels // 8) * 2, 3, padding=1, groups=4)
            self.BN_C9 = nn.BatchNorm2d((outchannels // 8) * 2)
            self.relu_C9 = nn.ReLU(inplace=False)

        #pooling
        self.pool2 = nn.AvgPool2d(2,2)

        #upsampling
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True)
        self.upscore8 = nn.Upsample(scale_factor=8, mode='bilinear',align_corners=True)

    def forward(self, x):

        score = x

        #part 1
        score1 = self.relu_C1(self.conv_C1(score))

        score1_1 = self.pool2(score1)
        score1_1 = self.relu_C2(self.BN_C2(self.conv_C2(score1_1)))

        score1_2 = self.pool2(score1_1)
        score1_2 = self.relu_C3(self.BN_C3(self.conv_C3(score1_2)))

        score1_3 = self.pool2(score1_2)
        score1_3 = self.relu_C4(self.BN_C4(self.conv_C4(score1_3)))

        t1 = self.upscore2(score1_1)
        t2 = self.upscore4(score1_2)
        t3 = self.upscore8(score1_3)

        score1 = self.relu_C5(self.BN_C5(self.conv_C5(score1 + t1 + t2 + t3)))
        #part 2
        score2 = self.relu_C6(self.conv_C6(score))

        score2_1 = self.pool2(score2)
        score2_1 = self.relu_DiC1(self.BN_DiC1(self.conv_DiC1(score2_1)))

        score2_2 = self.pool2(score2_1)
        score2_2 = self.relu_DiC2(self.BN_DiC2(self.conv_DiC2(score2_2)))

        score2_3 = self.pool2(score2_2)
        score2_3 = self.relu_DiC3(self.BN_DiC3(self.conv_DiC3(score2_3)))

        t1 = self.upscore2(score2_1)
        t2 = self.upscore4(score2_2)
        t3 = self.upscore8(score2_3)

        score2 = self.relu_C7(self.BN_C7(self.conv_C7(score2 + t1 + t2 + t3)))
        #part 3
        score3 = self.relu_C8(self.conv_C8(score))

        score3_1 = self.relu_DiC4(self.BN_DiC4(self.conv_DiC4(score3)))

        score3_2 = self.relu_DiC5(self.BN_DiC5(self.conv_DiC5(score3_1)))

        score3_3 = self.relu_DiC6(self.BN_DiC6(self.conv_DiC6(score3_2)))

        score3 = self.relu_C9(self.BN_C9(self.conv_C9(score3 + score3_1 + score3_2 + score3_3)))

        out = torch.cat((score1, score2, score3),1)

        return out