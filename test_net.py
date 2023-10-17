import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import time
from PIL import Image
from models.Corrnet.CorrNet_models import CorrelationModel_VGG
from data import test_dataset
from models.My.Mynet import *
# from models.ACCONet.ACCoNet_VGG_models import *
from models.EMFINet.EMFI_Net import *
from models.JRBM.CPD_models import *
from models.ERPNet.ERPNet_VGG import *
from models.FSMINet.FSMINet import *
from models.SeaNet.SeaNet_models import *

from models.NSI.PGNet.src.PGNet import *
from models.NSI.CPD.model.CPD_models import *
from models.NSI.BSANet.Src.BSANet import *
from models.NSI.GateNet.model_GateNet_vgg16 import *
from models.NSI.PFSNet.net import *
from models.NSI.SCRN.model.ResNet_models import *

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
opt = parser.parse_args()

dataset_path = r'E:\data\RSISOD\test/'
# dataset_path = r"E:\data\EORSSD\test/"
model = Net()
# model = nn.DataParallel(model)
model.load_state_dict(torch.load('./model/my_net/Net_epoch35_4119_vgg.pth'))

model.cuda()
model.eval()

test_datasets = ['RSISOD']
# test_datasets = ['BSO', 'CS', 'CSO', 'ISO', 'LSO', 'MSO', 'NSO', 'OC', 'SSO']

for dataset in test_datasets:
    save_path = './results/Ours/' + dataset + '/'
    rgb_savepath = r"E:\c_run\MyNet-main\results\Ours/RGB_ORSI4199/"
    if not os.path.exists(rgb_savepath):
        os.makedirs(rgb_savepath)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path  + '/image/'
    gt_root = dataset_path  + '/gt/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        time_start = time.time()
        output = model(image)
        time_end = time.time()
        time_sum = time_sum + (time_end - time_start)
        res = F.upsample(output[0], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        print('save img to: ', save_path + name)
        img_rgb = Image.fromarray(res * 255).convert('RGB')
        cv2.imwrite(save_path + name, res * 255)
        # img_rgb.save(rgb_savepath + name)
        if i == test_loader.size - 1:
            print('Running time {:.5f}'.format(time_sum / test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size / time_sum))
