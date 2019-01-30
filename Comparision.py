#-*- coding:gb18030 -*-
#thid code is for aid isprs journal


import sys
import os
import shutil
import csv
import subprocess
import random
import time
from os.path import walk
from PIL import Image
import numpy as np
from sklearn import calibration
from sklearn import svm
import caffe
import pdb
import datetime

imagesPath = "DataSet_JPG"

imagesWidth = 256
imagesHeight = 256
trainmethod = ""
labels = {
	'Airport': 0,
	'BareLand' : 1,
	'BaseballField': 2,
	'Beach': 3,
	'Bridge':4,
	'Center':5,
	'Church':6,
	'Commercial':7,
	'DenseResidential':8,
	'Desert':9,
	'Farmland':10,
	'Forest':11,
	'Industrial':12,
	'Meadow':13,
	'MediumResidential':14,
	'Mountain':15,
	'Park':16,
	'Parking':17,
	'Playground':18,
	'Pond':19,
	'Port':20,
	'RailwayStation':21,
	'Resort':22,
	'River':23,
	'School':24,
	'SparseResidential':25,
	'Square':26,
	'Stadium':27,
	'StorageTanks':28,
	'Viaduct':29,
}
caffe_bin = "build/tools/caffe"
caffe_convert_imageset = "build/tools/convert_imageset"
caffe_compute_image_mean = "build/tools/compute_image_mean"
caffe_path = "/home/hpc-126/caffe-master"
SVM_deployPath = ""
caffeModelPath = ""

def convert_image_and_convert_csv_v2(path):
	remove_file("csvfold/Test_.csv")
	remove_file("csvfold/Train_.csv")
	remove_file("csvfold/Unlabeled_.csv")
	remove_file("csvfold/Validation_.csv")
	images=[]
	imagesPerClass=[]
	train_sample,test_sample,validation_sample,unlabeled_sample=[],[],[],[]
	os.mkdir("DataSet_JPG")
	for root, dirs, files in os.walk(path):
		if root ==path:
			continue

		category =os.path.basename(root)
		label = labels[category]
#		pdb.set_trace()
		for name in files:
			im = Image.open(os.path.join(root, name))
			(width, height) = im.size
			# images in the UCMerced_LandUse dataset are supposed to be 256x256, but they aren't
			if width != imagesWidth or height != imagesHeight:
				im = im.resize((imagesWidth, imagesHeight), Image.ANTIALIAS)

			jpeg_name = name.replace(".tif", ".jpg")

			im.save(os.path.join(imagesPath, jpeg_name))

			images.append([ jpeg_name, label ])
			imagesPerClass.append([jpeg_name, label])

		random.shuffle(imagesPerClass)
		numfile= len(imagesPerClass)
		
		for i in range (0,int(0.2*numfile)):
			test_sample.append(imagesPerClass[i])

	
		for i in range (int(0.4*numfile),int(numfile*0.5)):
			train_sample.append(imagesPerClass[i])
	
		for i in range (int(0.5*numfile),numfile):
			unlabeled_sample.append(imagesPerClass[i])

	
		for i in range (int(0.2*numfile),int(0.4*numfile)):
			validation_sample.append(imagesPerClass[i])

		imagesPerClass=[]

	random.shuffle(images)
	random.shuffle(test_sample)
	random.shuffle(train_sample)
	random.shuffle(validation_sample)
	random.shuffle(unlabeled_sample)
	with open("csvfold/Test_.csv", "a") as csvFile:
		csvWriter = csv.writer(csvFile, delimiter=' ')

		for image in test_sample:
			csvWriter.writerow(image)

	with open("csvfold/Train_.csv", "a") as csvFile:
		csvWriter = csv.writer(csvFile, delimiter=' ')
		for image in train_sample:
			csvWriter.writerow(image)
	with open("csvfold/Unlabeled_.csv", "a") as csvFile:
		csvWriter = csv.writer(csvFile, delimiter=' ')
		for image in unlabeled_sample:
			csvWriter.writerow(image)

	with open("csvfold/Validation_.csv","a")as csvFile:
		csvWriter = csv.writer(csvFile, delimiter=' ')
		for image in validation_sample:
			csvWriter.writerow(image)	
def run_command(command):
    print("Running: " + ' '.join(command))
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    lines = iter(p.stdout.readline, b'')

    data = []
    for line in lines:
      print(line)
      data.append(line)

    while True:
      p.poll()
      if p.returncode == None:
        time.sleep(1)
      elif p.returncode != 0:
        raise Exception("Error while running command: " + str(p.returncode))
      else:
        break

    return data


def remove_dir(path):
    try:
      shutil.rmtree(path)
    except OSError, e:
      if e.errno == 2:
        pass
      else:
        raise

def remove_file(path):
    try:
      os.unlink(path)
    except OSError, e:
      if e.errno == 2:
        pass
      else:
        raise

def classify(net, files, oversample=True):
    images = []

    for file in files:
      images.append(caffe.io.load_image(os.path.join(imagesPath, file)))

    return net.predict(images, oversample)




def svmTrain(num):

	caffe.set_mode_gpu()

	net = caffe.Classifier(SVM_deployPath,
	                     caffeModelPath,
	                     mean=np.load(os.path.join(caffe_path, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')).mean(1).mean(1),
	                     channel_swap=(2,1,0),
	                     raw_scale=255,
	                     image_dims=(imagesWidth, imagesHeight))
	starttime=datetime.datetime.now()
	X=[]
	y=[]
	np.array(X)
	np.array(y)
	with open("csvfold/Train_.csv", "rb") as csvFile:
		csvReader = csv.reader(csvFile, delimiter=' ')
		count=0
		i=0
		for row in csvReader:
		  prediction= classify(net,  [ row[0] ])
		#      prediction = classify(net, [ row[0] ])
		  X.append(np.array(prediction[0]))
		  y.append(np.array(row[1]))
		  i+=1
		  count+=1
		  #count=16800
		  if i==100 or count==420:
		  	print("has train samples:"+ str(count))
		  	i=0

	#use to test code
	#      if count==1680:
	#      	break

	a=len(prediction[0])
	print("finish train process")

	clf = svm.LinearSVC(C=100000,max_iter=5000)
	clf.fit(X, y)
	correct = 0
	total = 0
	wrong = 0
	print("start test process")
	with open("csvfold/Test_.csv", "rb") as csvFile:
		csvReader = csv.reader(csvFile, delimiter=' ')
		for row in csvReader:
			features = classify(net, [ row[0] ])
			#      features = classify(net,imagesPath, [ row[0] ])
			#features=get_average_vector(net,data, [ row[0] ],a)
			#      prediction = clf.predict(features[0].reshape(1,-1))
			prediction = clf.predict(np.array(features).reshape(1, a))
			if prediction == row[1]:
				correct += 1
			else:
				wrong += 1
			total += 1
	print("TOTAL: " + str(total))
	print("CORRECT: " + str(correct))
	print("WRONG: " + str(wrong))
	endtime=datetime.datetime.now()
	computing_time=(endtime- starttime).seconds
	print("the computation time:"+str(computing_time))
	output = trainmethod+"\n"+  "TOTAL: " + str(total)+"\n"+"CORRECT: " + str(correct)+"\n"+"WRONG: " + str(wrong)+"\n"+"accuracy: " + str(float(correct) / float(total) * 100)+"\n"+"the computation time:"+str(computing_time) + "\n\n"
	open("results_10train.txt", "a").write(output + "\n")

def svmTrain2(num):


	caffe.set_mode_gpu()

	net = caffe.Classifier(SVM_deployPath,
	                     caffeModelPath,
	                     mean=np.load(os.path.join(caffe_path, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')).mean(1).mean(1),
	                     channel_swap=(2,1,0),
	                     raw_scale=255,
	                     image_dims=(imagesWidth, imagesHeight))

	starttime=datetime.datetime.now()
	X=[]
	y=[]
	np.array(X)
	np.array(y)
	with open("csvfold/Train_.csv", "rb") as csvFile:
		csvReader = csv.reader(csvFile, delimiter=' ')
		count=0
		i=0
		for row in csvReader:
		  prediction= classify(net, [ row[0] ])
		#      prediction = classify(net, [ row[0] ])
		  X.append(np.array(prediction[0]))
		  y.append(np.array(row[1]))
		  i+=1
		  count+=1
		  #count=16800
		  if i==100 or count==420:
		  	print("has train samples:"+ str(count))
		  	i=0
	with open("csvfold/Unlabeled_.csv", "rb") as csvFile:
		csvReader = csv.reader(csvFile, delimiter=' ')
		count=0
		i=0
		for row in csvReader:
		  prediction= classify(net, [ row[0] ])
		#      prediction = classify(net, [ row[0] ])
		  X.append(np.array(prediction[0]))
		  y.append(np.array(row[1]))
		  i+=1
		  count+=1
		  #count=16800
		  if i==100 or count==840:
		  	print("has train samples:"+ str(count))
		  	i=0
	#use to test code
	#      if count==1680:
	#      	break

	a=len(prediction[0])
	print("finish train process")

	clf = svm.LinearSVC(C=100000,max_iter=5000)
	clf.fit(X, y)
	correct = 0
	total = 0
	wrong = 0
	print("start test process")
	with open("csvfold/Test_.csv", "rb") as csvFile:
		csvReader = csv.reader(csvFile, delimiter=' ')
		for row in csvReader:
			features = classify(net, [ row[0] ])
			#      features = classify(net,imagesPath, [ row[0] ])
			#features=get_average_vector(net,data, [ row[0] ],a)
			#      prediction = clf.predict(features[0].reshape(1,-1))
			prediction = clf.predict(np.array(features).reshape(1, a))
			if prediction == row[1]:
				correct += 1
			else:
				wrong += 1
			total += 1
	print("TOTAL: " + str(total))
	print("CORRECT: " + str(correct))
	print("WRONG: " + str(wrong))
	endtime=datetime.datetime.now()
	computing_time=(endtime- starttime).seconds
	print("the computation time:"+str(computing_time))
	output = trainmethod+"\n"+  "TOTAL: " + str(total)+"\n"+"CORRECT: " + str(correct)+"\n"+"WRONG: " + str(wrong)+"\n"+"accuracy: " + str(float(correct) / float(total) * 100)+"\n"+"the computation time:"+str(computing_time) + "\n\n"
	open("results_60train.txt", "a").write(output + "\n")



def go(num):
	global SVM_deployPath,caffeModelPath,imagesHeight,imagesWidth,trainmethod

	if trainmethod=="all":
		remove_dir("DataSet_JPG")
		convert_image_and_convert_csv_v2("DataSet")
		#generate the image data for 227x227
		trainmethod ="caffenet"
		SVM_deployPath = "deploy_svm_caffenet.prototxt"
		caffeModelPath = "/home/hpc-126/caffe-master/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
		svmTrain(num)
		svmTrain2(num)

		# trainmethod ="Placenet"
		# SVM_deployPath = "/home/hpc-126/caffe-master/models/placesCNN/placesnet_deploy_svm.prototxt"
		# caffeModelPath = "/home/hpc-126/caffe-master/models/placesCNN/places205CNN_iter_300000.caffemodel"
		# svmTrain(num)


		#generate the image data for 224x224
		

		trainmethod ="googlenet"
		SVM_deployPath ="deploy_googlenet_svm.prototxt"
		caffeModelPath = os.path.join(caffe_path, "models", "bvlc_googlenet", "bvlc_googlenet.caffemodel")
		svmTrain(num)
		svmTrain2(num)

		trainmethod="CNN_F"
		SVM_deployPath = "/home/hpc-126/caffe-master/models/VGG_F/VGG_CNN_F_deploy_svm.prototxt"
		caffeModelPath = "/home/hpc-126/caffe-master/models/VGG_F/VGG_CNN_F.caffemodel"
		svmTrain(num)
		svmTrain2(num)


		trainmethod = "CNN_S"
		SVM_deployPath = "/home/hpc-126/caffe-master/models/VGG_S/VGG_CNN_S_deploy_svm.prototxt"
		caffeModelPath = "/home/hpc-126/caffe-master/models/VGG_S/VGG_CNN_S.caffemodel"
		svmTrain(num)
		svmTrain2(num)
		trainmethod= "CNN_M"
		SVM_deployPath = "/home/hpc-126/caffe-master/models/VGG_M/VGG_CNN_M_deploy_svm.prototxt"
		caffeModelPath = "/home/hpc-126/caffe-master/models/VGG_M/VGG_CNN_M.caffemodel"
		svmTrain(num)
		svmTrain2(num)
		trainmethod = "VGG_16"
		SVM_deployPath = "/home/hpc-126/caffe-master/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers_deploy_svm.prototxt"
		caffeModelPath = "/home/hpc-126/caffe-master/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel"
		svmTrain(num)
		svmTrain2(num)
		trainmethod ="VGG_19"
		SVM_deployPath = "/home/hpc-126/caffe-master/models/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers_deploy_svm.prototxt"
		caffeModelPath = "/home/hpc-126/caffe-master/models/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.caffemodel"
		svmTrain(num)
		svmTrain2(num)
		trainmethod ="ResNet_50"
		SVM_deployPath ="/home/hpc-126/caffe-master/models/resnet_50/delpoy_svm.prototxt"
		caffeModelPath ="/home/hpc-126/caffe-master/models/resnet_50/ResNet-50-model.caffemodel"
		svmTrain(num)
		svmTrain2(num)
	trainmethod="all"





def main (argv):
	
    for num in range (1,7):

#        output=str(num)+"   generate data!!!"
#        open("results.txt", "a").write("whether use aug is :" + str(is_not_use_augmentation) + "\n" + output + "\n")
		open("results_10train.txt","a").write(" 10 percentes trainset condition, the test order is: "+ str(num)+", the results are as followed")
		open("results_60train.txt","a").write(" 60 percentes trainset condition, the test order is: "+ str(num)+", the results are as followed")
		go(num)



if __name__ == "__main__":
  main(sys.argv)
