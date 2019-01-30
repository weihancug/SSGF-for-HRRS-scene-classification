# SSGF-for-HRRS-scene-classification
##experimental codes for "A semi-supervised generative framework with deep learning features for high-resolution remote sensing image scene classification"

##Datasets
The datasets used in the work is UCM, WHU-RS19, NWPU-RESISC45, and AID
(a)UCM
![UCM](https://github.com/weihancug/SSGF-for-HRRS-scene-classification/blob/master/UCM-dataset.png)
(b)WHU-RS19
![WHU-RS19](https://github.com/weihancug/SSGF-for-HRRS-scene-classification/blob/master/WHU-dataset.png)
(c)NWPU-RESISC45
![NWPU](https://github.com/weihancug/SSGF-for-HRRS-scene-classification/blob/master/NUPW-45.png)
(d)
![AID](https://github.com/weihancug/SSGF-for-HRRS-scene-classification/blob/master/AID-dataset.png)

##Illustration of code
(a)selftraining.py is the implenmtation of selftraining method by using one pre-trained CNN model
(b)co-training.py is the cotraining method, wherein two kind of deep features are utilized.
(c)Comparision.py is the comparison methods.
(d)plot_confusion_matrix.py is to draw the confusion matrix.
