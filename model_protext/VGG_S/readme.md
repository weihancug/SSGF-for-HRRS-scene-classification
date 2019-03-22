##Information

name: CNN_S model from the BMVC-2014 paper: "Return of the Devil in the Details: Delving Deep into Convolutional Nets"

mean_file_mat: http://www.robots.ox.ac.uk/~vgg/software/deep_eval/releases/bvlc/VGG_mean.mat

mean_file_proto: http://www.robots.ox.ac.uk/~vgg/software/deep_eval/releases/bvlc/VGG_mean.binaryproto

caffemodel: VGG_CNN_S

caffemodel_url: http://www.robots.ox.ac.uk/~vgg/software/deep_eval/releases/bvlc/VGG_CNN_S.caffemodel

license: non-commercial use only

caffe_version: trained using a custom Caffe-based framework

gist_id: fd8800eeb36e276cd6f9

## Description

The CNN_S model is trained on the ILSVRC-2012 dataset. The details can be found in the following [BMVC-2014 paper](http://www.robots.ox.ac.uk/~vgg/publications/2014/Chatfield14/):

    Return of the Devil in the Details: Delving Deep into Convolutional Nets
    K. Chatfield, K. Simonyan, A. Vedaldi, A. Zisserman
    British Machine Vision Conference, 2014 (arXiv ref. cs1405.3531)

Please cite the paper if you use the model.

The model is trained on 224x224 crops sampled from images, rescaled so that the smallest side is 256 (preserving the aspect ratio). The released mean BGR image should be subtracted from 224x224 crops.

Further details can be found in the paper and on the project website: http://www.robots.ox.ac.uk/~vgg/research/deep_eval/

#### Note

The model is stored in a different format than the one released at http://www.robots.ox.ac.uk/~vgg/software/deep_eval/ to make it compatible with BVLC Caffe and BGR images (the network weights are the same). The class order is also different; the one used here corresponds to *synsets.txt* in http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz

## ILSVRC-2012 performance

Using 10 test crops (corners, centre, and horizontal flips), the top-5 classification error on the validation set of ILSVRC-2012 is 13.1%.

Using a single central crop, the top-5 classification error on the validation set of ILSVRC-2012 is 15.4%

The details of the evaluation can be found in the paper.