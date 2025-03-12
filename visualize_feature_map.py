# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 00:58:20 2022

@author: Chen'hong'yu
"""
import matplotlib.pyplot as plt
import torchvision
from pylab import *
import seaborn as sns
from PIL import Image


# from models.FCN import *

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(feature_map, num_pic):
    feature_map = feature_map.detach().numpy()
    # feature_map =  torch.squeeze(feature_map, 0)
    # feature_map = 0.8 * (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map)) + 0.1
    # print(feature_map)
    feature_map_combination = []
    plt.figure()

    row, col = get_row_col(num_pic)

    for i in range(0, num_pic):
        feature_map_split = feature_map[i, :, :]
        ''' 通道维度'''
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split, cmap="jet")  # cmap="jet" 代表热图
        # plt.colorbar()
        # plt.imshow(feature_map_split,)
        axis('off')  # 去掉坐标轴
    # plt.savefig(r"C:\Users\Chen'hong'yu\Desktop\BPC/DFFC_after.png", bbox_inches='tight', dpi=300, transparent=True)
    plt.show()

    '''各个特征图按1：1 叠加 空间维度'''

    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum, cmap="jet")
    # plt.colorbar()
    # plt.imshow(feature_map_sum ,)
    axis('off')
    ### 加边框
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.savefig(r"./ss.png", bbox_inches='tight', dpi=300, transparent=True)
    plt.show()


    '''保存想要的特征图'''
    # want = feature_map_combination[13]
    # want = sum(ele for ele in feature_map_combination)
    # plt.imshow(want,cmap="jet")
    # axis('off')
    # plt.savefig("FCNbri_jet.png", bbox_inches='tight', dpi=300, transparent=True)


if __name__ == "__main__":
    # from Net_1_j3gaie4 import *
    # from baseline.Unet1 import *
    import os
    from new_SOD_pvt_gai_merge_decoder_SOTA import *
    # from models.My.MCCNet_models import *
    import cv2

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                             [0.229, 0.224, 0.225])
                                            ])
    # img = cv2.imread(r"E:\data\WHU_BUILDING\test\image/620.tif")
    # img_1 = cv2.imread(r'E:\data\EORSSD\test\image/0022.jpg')
    # img_1 = Image.open(r"E:\data\SOD\EORSSD\test\image/0823.jpg").convert('RGB')
    imgs = os.listdir(r"E:\data\SOD\EORSSD\test\image/")
    module = r'E:\c_run\New_SOD\model\my_net/Net_epoch14_best_EORSSD_0.004612626898042016.pth'

    for item in imgs:
        img_name = os.path.basename(item)
        img_path = os.path.join(r"E:\data\SOD\EORSSD\test\image/", item)
        img_1 = Image.open(img_path).convert('RGB')
        if img_1.size[0] != 352 or img_1.size[1] != 352:
            img = img_1.resize((352, 352))
        else:
            img = img_1
        img_tensor = trans(img).unsqueeze(0)
        model = SOD()
        model.load_state_dict(torch.load(module))
        model.eval()
        conv_img = model(img_tensor)
        if isinstance(conv_img, (list, tuple)):
            conv_img = F.upsample(conv_img[0], size=img_1.size, mode='bilinear', align_corners=False)
            conv_img = conv_img.squeeze(0)
        else:
            conv_img = conv_img.squeeze(0)
        # print(conv_img[0])
        print(conv_img.shape)
        visualize_feature_map(conv_img, num_pic=conv_img.shape[0])




    # module = r'F:\ww\dataset_gao\res\SCNN/Epoch_15_Loss_0.266_Acc_0.973_Iou_0.883_nodf.pkl'

    # module = r"E:\c_run\MyNet-main\model\ERPNet/MYNet_126000_89_EORSSD.pth"
    # module = r'E:\c_run\MCCNet-main\model\MCC/mccNet_epoch_34_ORSSD.pth'
    # module = r'F:\ww\dataset_gao\res\SCNN/Epoch_42_Loss_0.811_Acc_0.972_Iou_0.880_nostc.pkl'
    # module = r'F:\ww\dataset_gao\res\SCNN/Epoch_15_Loss_0.266_Acc_0.973_Iou_0.883_nodf.pkl'




    # vgg_model = VGGNet(requires_grad=True)
    # model = FCN32s(pretrained_net=vgg_model, n_class=1)
    # model = ERPNet_VGG()

    '''num_pic=conv_img.shape[0] 代表显示当前层输出的通道数量张特征图
    可以自己指定显示前几张特征图，但是数量不能超过当前输出的通道数'''

