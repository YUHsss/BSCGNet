# -*- coding: utf-8 -*-
"""
Created on Tue May  3 22:47:37 2022

@author: Chen'hong'yu
"""

import time

import torch.backends.cudnn as cudnn

import tqdm
import warnings
import time
import torch
from models.My.Mynet import *
from models.My.MCCNet_models import *
from models.Corrnet.CorrNet_models import *
from models.ACCONet.ACCoNet_VGG_models import *
from models.EMFINet.EMFI_Net import *
from models.JRBM.CPD_models import *
from models.ERPNet.ERPNet_VGG import *
from models.FSMINet.FSMINet import *
from models.NL.CPD.model.CPD_models import *
from models.NL.GateNet.model_GateNet_vgg16 import *
from models.NL.SCRN.model.ResNet_models import *
from models.NL.AGNet.res2net import AGNet
from models.NL.BSANet.Src.BSANet import *
from models.NL.PFSNet.net import *
import numpy as np
from models.NL.CTDNet.src.net import *
warnings.filterwarnings('ignore')

def compute_speed(model, input_size, iter):
    device = 'cuda:0'
    cudnn.benchmark = True  # 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    model.eval()
    model = model.to(device)
    input = torch.randn(input_size).to(device)

    print('---------开始GPU预热---------')
    with torch.no_grad():
        for _ in tqdm.tqdm(range(100)):
            model(input)

    print('---------预热结束，计算模型推理时间中....---------')
    torch.cuda.synchronize()  # 增加同步操作，等待GPU完成操作，更加准确测试模型速度
    t_start = time.time()
    for _ in tqdm.tqdm(range(iter)):
        model(input)
    torch.cuda.synchronize()
    run_time = time.time() - t_start

    speed_time = run_time / iter * 1000
    fps = iter / run_time

    print('运行时间: [%.2f s / %d iter]' % (run_time, iter))
    print('迭代一次时间: %.2f ms / iter\t\nFPS: %.2f' % (speed_time, fps))
    return speed_time, fps
if __name__ == '__main__':
    model = Net()
    input_size = (1, 3, 256, 256)
    compute_speed(model, input_size, iter=200)




