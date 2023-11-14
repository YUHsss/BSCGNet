<p align="center">

  <h3 align="center">Boundary-Semantic Collaborative Guidance Network with dual-stream feedback mechanism for Salient Object Detection in Optical Remote Sensing Imagery</h3>

  <p align="center">
   


# Network Architecture
   <div align=center>
   <img src=https://github.com/YUHsss/BSCGNet/blob/main/image/BSCGNet.jpg>
   </div>
   
# Accuracy
   <div align=center>
   <img src=https://github.com/YUHsss/BSCGNet/blob/main/image/table.jpg>
   </div> 
   
# Saliency maps
    We provide saliency maps of our BSCGNet on ORSSD, EORSSD and ORSI-4199 in './saliency_maps/Ours.7z'.
    
# Training
Backbone: [VGG16](https://pan.baidu.com/s/1OF5tn5qqmgXRFf71HCvXaQ) (code: 7a1s)

Our BSCGNet can be found in './models/BSCGNet/Mynet.py'

Modify paths of datasets, then run train_net.py.

# Testing
modify paths of pre-trained model and datasets, then run test_net.py.

# Evaluation Tool
   run evaluation.py to evaluate the above saliency maps.
   
# Related Works
    we provide a common code base which has 17 SOD methods, including 8 ORSI-SOD methods, 7 NSI-SOD methods, 2 traditional SOD methods and the method proposed in this
    paper.You can find them directly in './models/'.
    
The references are as follows.
* Traditional methods

DSG: (TIP 2017) Salient region detection using diffusion process on a two-layer sparse graph

RCNN: (TIP 2018) Reversion Correction and Regularized Random Walk Ranking for Saliency Detection
* DL-based NSI-SOD

SCRN: (ICCV 2019) Stacked cross refinement network for edge-aware salient object detection

CPDNet: (CVPR 2019) Cascaded partial decoder for fast and accurate salient object detection

GateNet: (ECCV 2020) Suppress and balance: A simple gated network for salient object detection

CTDNet: (ACMM 2021) Complementary trilateral decoder for fast and accurate salient object detection

PFSNet: (AAAI 2021) Pyramidal feature shrinking for salient object detection

BSANet: (AAAI 2022) I can find you! Boundary-guided Separated Attention Network for Camouflaged Object Detection

PGNet: (CVPR 2022) Pyramid Grafting Network for One-Stage High Resolution Saliency Detection

* DL-based ORSI-SOD

LVNet: (TGRS 2019) Nested network with two-stream pyramid for salient object detection in optical remote sensing images. 

FSMINet: (GRSL 2022) Fully Squeezed Multiscale Inference Network for Fast and Accurate Saliency Detection in Optical Remote-Sensing Images

ERPNet: (TCYB 2022) Edge-guided recurrent positioning network for salient object detection in optical remote sensing images

CorrNet: (TGRS 2022) Lightweight salient object detection in optical remote sensing images via feature correlation

SeaNet: (TGRS 2023) Lightweight Salient Object Detection in Optical Remote-Sensing Images via Semantic Matching and Edge Alignment

SRAL: (TGRS 2023) Distilling Knowledge From Super-Resolution for Efficient Remote Sensing Salient Object Detection

EMFINet: (TGRS 2022) Edge-aware multiscale feature integration network for salient object detection in optical remote sensing images

JRBM: (TGRS 2022) ORSI salient object detection via multiscale joint region and boundary model

# Citation


If you encounter any problems with the code.

Please contact me at chy0519@my.swjtu.edu.cn.
