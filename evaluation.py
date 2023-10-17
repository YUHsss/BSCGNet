# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
from tqdm import tqdm
from metrics.sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

# method = 'My'
for _data_name in ["EORSSD"]:
    model_name = ""
    mask_root = r''
    pred_root = os.path.join(r"E:\c_run\MyNet-main\results",model_name,_data_name)
    # pred_root = r'E:\c_run\MyNet-main\results\ACCoNet\EORSSD/'
    mask_name_list = sorted(os.listdir(mask_root))
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)

    fm = FM.get_results(model_name,_data_name)["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results(model_name,_data_name)["sm"]
    em = EM.get_results(model_name,_data_name)["em"]
    mae = M.get_results(model_name,_data_name)["mae"]

    results = {
        "MAE": round(mae, 6),
        "Smeasure": round(sm, 6),
        # "wFmeasure": round(wfm, 6),
        "adpEm": round(em["adp"], 6),
        "meanEm": round(em["curve"].mean(), 6),
        "maxEm": round(em["curve"].max(), 6),
        "adpFm": round(fm["adp"], 6),
        "meanFm": round(fm["curve"].mean(), 6),
        "maxFm": round(fm["curve"].max(), 6),
    }

    print(results)
    # file = open("evalresults.txt", "a")
    # file.write(method + ' ' + _data_name + ' ' + str(results) + '\n')
