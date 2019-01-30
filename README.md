# SSGF-for-HRRS-scene-classification
Code to replicate the analyses from the paper [A semi-supervised generative framework with deep learning features for high-resolution remote sensing image scene classification](https://www.sciencedirect.com/science/article/abs/pii/S0924271617303428)
 
Datasets
===================================  
  
The datasets used in the work is UCM, WHU-RS19, NWPU-RESISC45, and AID
-----------------------------------  
1 UCM
![UCM](https://github.com/weihancug/SSGF-for-HRRS-scene-classification/blob/master/UCM-dataset.png)
2 WHU-RS19
![WHU-RS19](https://github.com/weihancug/SSGF-for-HRRS-scene-classification/blob/master/WHU-dataset.png)
3 NWPU-RESISC45
![NWPU](https://github.com/weihancug/SSGF-for-HRRS-scene-classification/blob/master/NUPW-45.png)
4 AID
![AID](https://github.com/weihancug/SSGF-for-HRRS-scene-classification/blob/master/AID-dataset.png)

 
Illustration of code
===================================  

1. selftraining.py is the implenmtation of selftraining method by using one pre-trained CNN model.

2. co-training.py is the cotraining method, wherein two kind of deep features are utilized.

3. Comparision.py is the comparison methods.

4. plot_confusion_matrix.py is to draw the confusion matrix.

