
# One-Class Classifier for Chest X-ray Anomaly Detection via ContrastivePatch-based Percentile

Chest X-ray anomaly detection via patch-wise percentile
We release CSIP image test code.


Detailed instructions for testing the image are as follows.

------

# Implementation

A PyTorch implementation of One-Class Classifier for Chest X-ray Anomaly Detection via ContrastivePatch-based Percentile based on original CSI and Deep-Learning-COVID-19-on-CXR-using-Limited-Training-Data-Sets code.

CSI[https://github.com/alinlab/CSI] (*Thanks for Jihoon Tack*, Sangwoo Mo*, Jongheon Jeong, and Jinwoo Shin.)
Deep-Learning-COVID-19-on-CXR-using-Limited-Training-Data-Sets


[https://github.com/jongcye/Deep-Learning-COVID-19-on-CXR-using-Limited-Training-Data-Sets](*Thanks for Yujin Oh*, Sangjoon Park*, and Jong Chul Ye)

------
## Environments

The setting of the virtual environment we used is described as CSIP.yaml.

------
## Segmentation

Put the test image in the "inputs" folder to create a split mask. After that, receive the segmentation model pre-trained weight file above and put it in "segmentation/segmentation_checkpoint". Please run "segmentation/codes/inference.py".

python inference.py 
The segmented mask (same name_mask.jpg) and the preprocessed image (same name) are stored in the "input" folder.

Download segmentation_checkpoint file(segmentation_model) in [click](https://drive.google.com/drive/folders/1WvvwwY3O9ItcZ8G6Y71D3g3GWK0mNsSW?usp=sharing) 

------
## CSIP

Download last.model, train_shift_features.pth, train_simclr_features.pth from the link below in "CSIP/weight_folder" before performing "CSIP". Codes/run_local.Run sh.

bash run_local.sh

Download CSIP weight_folder file(last.model, train_shift_features.pth, train_simclr_features.pth) in [click](https://drive.google.com/drive/folders/1GBM8zIFwYi0OodXLenJQDCGF6VNELF80?usp=sharing)

------
## Result

If you run two things sequentially, you will see that a "visual" folder is created in the "CSIP" folder, storing the gradcam image and outputting statistical indicators.

------





