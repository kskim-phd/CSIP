
# One-Class Classifier for Chest X-rayAnomaly Detection via ContrastivePatch-based Percentile


Chest X-ray anomaly detection via patch-wise percentile



Detailed instructions for testing the image are as follows.


------
# Segmentation

Put the test image in the "inputs" folder to create a split mask. After that, receive the segmentation model pre-trained weight file above and put it in "segmentation/segmentation_checkpoint". Please run "segmentation/codes/inference.py".

The segmented mask (same name_mask.jpg) and the preprocessed image (same name) are stored in the "input" folder.

Download segmentation_checkpoint file(segmentation_model) in https://drive.google.com/drive/folders/1WvvwwY3O9ItcZ8G6Y71D3g3GWK0mNsSW?usp=sharing

------
# CSIP

Download CSIP weight_folder file(last.model, train_shift_features.pth, train_simclr_features.pth) in https://drive.google.com/drive/folders/1GBM8zIFwYi0OodXLenJQDCGF6VNELF80?usp=sharing
