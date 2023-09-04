# HEN：
A Deep Learning-based Hybrid Encoder Network for Flood Image Segmentation
# Installation Requirement:
1.	Win10 and above or Ubuntu can be used for this experiment.
2.	Installing python development software (e.g.: Pycharm ,VS code).
3.	PC configuration NVIDIA GeForce RTX 2080 and above.
4.	Requires Python 3.7 and above.
5.	Requires configuration of CUDA 11.5 and above.
# Usage：
1.  You can check the image format. If the image format isn’t .png, convert.py can be used to convert the image format.
2.  In the config.py, you can change image dimensions, training parameters, dataset paths, model save path, and VIT parameters.
3.  train.py is used to train the data. There are five items in this project named after 'model name.py ‘. If you want to replace them, you can replace them directly in 'Model _ name = [] ' in train.py.
4.  After training, the weight files of each model are put into eval _.py to verify the model and draw the comparison chart of evaluation indexes.
5.  Using show _ pic.py for predictive image processing.
6.  Draw the curve of train loss, train acc, val acc by show _ pic.py.
