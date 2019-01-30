# version 2 two classfier used here vote to desicion the classification result

#confusion matrix is painted here 

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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


imagesPath = "DataSet_JPG"
imageWidth = 256
imageHeight = 256
trainmethod = "caffenet"

labels = {
  'golfcourse': 0,
  'overpass': 1,
  'freeway': 2,
  'denseresidential': 3,
  'mediumresidential': 4,
  'harbor': 5,
  'tenniscourt': 6,
  'mobilehomepark': 7,
  'parkinglot': 8,
  'agricultural': 9,
  'chaparral': 10,
  'airplane': 11,
  'river': 12,
  'baseballdiamond': 13,
  'intersection': 14,
  'beach': 15,
  'runway': 16,
  'forest': 17,
  'sparseresidential': 18,
  'buildings': 19,
  'storagetanks': 20
}
caffe_bin = "build/tools/caffe"
caffe_convert_imageset = "build/tools/convert_imageset"
caffe_compute_image_mean = "build/tools/compute_image_mean"
caffe_path = "/home/hpc-126/caffe-master"
SVM_deployPath = ""
caffeModelPath = ""
SVM_deployPath2 = ""
caffeModelPath2 = ""

def convert_images(path):
  images = []
#  os.mkdir("DataSet_JPG")
  for root, dirs, files in os.walk(path):
    if root == path:
      continue

    category = os.path.basename(root)
    label = labels[category]

    for name in files:
      im = Image.open(os.path.join(root, name))
      (width, height) = im.size
# images in the UCMerced_LandUse dataset are supposed to be 256x256, but they aren't
      if width != imageWidth or height != imageHeight:
        im = im.resize((imageWidth, imageHeight), Image.ANTIALIAS)

      jpeg_name = name.replace(".tif", ".jpg")

      im.save(os.path.join(imagesPath, jpeg_name))

      images.append([ jpeg_name, label ])
      random.shuffle(images)
  return images


def convert_data_to_csv(images):
	os.mkdir("csvfold")
	for i in range(1,11):
		start_set = 210 * (i-1)
		end_set = 210 * i

		test_set = images[start_set:end_set]
		train_set = images[:start_set] + images[end_set:]

		with open("csvfold/Test_" + str(i) + ".csv", "wb") as csvFile:
			csvWriter = csv.writer(csvFile, delimiter=' ')

			for image in test_set:
			  csvWriter.writerow(image)

		with open("csvfold/Train_" + str(i) + ".csv", "wb") as csvFile:
			csvWriter = csv.writer(csvFile, delimiter=' ')

			for image in train_set:
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
#    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=18)
    plt.yticks(tick_marks, classes, rotation=30, fontsize=18)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
 #   else:
#        print('Confusion matrix, without normalization')

 #   print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		if normalize:
		    plt.text(j, i, round(cm[i, j],4),
		             horizontalalignment="center",
		             color="white" if cm[i, j] > thresh else "black")
		else:
			plt.text(j, i, cm[i, j],
		             horizontalalignment="center",
		             color="white" if cm[i, j] > thresh else "black")


	plt.tight_layout()
	plt.ylabel('True label',fontsize=25)
	plt.xlabel('Predicted label',fontsize=25)



def plot_save_graph(y_test, y_pred,j,claName):
	cnf_matrix = confusion_matrix(y_test, y_pred)
	np.set_printoptions(precision=2)
#	plt.figure(figsize=(20,))
	# plot_confusion_matrix(cnf_matrix, classes=claName,
	#                       title='Confusion matrix, without normalization')
	# plt.savefig("confusion_matrix/cnf_"+str(j)+".png")

	plt.figure(figsize=(22,20))
	plot_confusion_matrix(cnf_matrix, classes=claName, normalize = True,
						title= 'Confusion matrix, wioutnorimalization')
#	plt.show()
	plt.savefig("confusion_matrix/cnf_nor_"+str(j)+".png")


def svmTrain(i):
#   seg_ratio is the ratio between labeled samples and unlabeled samples
#   number_iteration 
	seg_ratio=1
	MaxNumPerClassPerIteration=10
# to 
	correct = 0
	total = 0
	wrong = 0	
	correctPerClass = [0 for k in range(21)]
	wrongPerClass = [0 for k in range(21)]
	numPerClass = [0 for k in range(21)]	
	
	addedsamplenum=0
# caffe construct the net
# 
	caffe.set_mode_gpu()
	net = caffe.Classifier(SVM_deployPath,
                         caffeModelPath,
                         mean=np.load(os.path.join(caffe_path, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')).mean(1).mean(1),
                         channel_swap=(2,1,0),
                         raw_scale=255,
                         image_dims=(256, 256))

	net2 = caffe.Classifier(SVM_deployPath2,
                         caffeModelPath2,
                         mean=np.load(os.path.join(caffe_path, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')).mean(1).mean(1),
                         channel_swap=(2,1,0),
                         raw_scale=255,
                         image_dims=(256, 256))

	X = []
	y = []
	samples=[]
	with open("csvfold/Train_" + str(i) + ".csv", "rb") as csvFile:
		csvReader = csv.reader(csvFile, delimiter=' ')
		for row in csvReader:
			samples.append(row[0])
			y.append(row[1])

#divided the dataset into labeled and unlabeled section according the ratio seg_ratio
	labeled_sample, labeled_y= samples[:210*seg_ratio], y[:210*seg_ratio]
	unlabeled_sample, unlabeled_y=samples[210*seg_ratio:],y[210*seg_ratio :]

	#use EL to save the learnt result
	EL_samples, EL_y = [],[]
	j=1
	train_sample, train_y=labeled_sample, labeled_y	

#train_X is the vector collection of CNN1 and train_X2 is the vector collection of CNN2, in this way clf2 is the classifier of CNN2

	train_X,train_X2=[],[]
	lowConfidence_sample, lowConfidence_y=[],[]

	for k in train_sample:
		prediction = classify(net,[k])
		train_X.append(prediction[0])
		prediction = classify(net2,[k])
		train_X2.append(prediction[0])
	#train svm
	lenofvector = len(train_X[0])
	lenofvector2 = len(train_X2[0])
	clf = calibration.CalibratedClassifierCV(svm.LinearSVC(C=100000))
	clf.fit(train_X,train_y)

	clf2 = calibration.CalibratedClassifierCV(svm.LinearSVC(C=100000))
	clf2.fit(train_X2,train_y)
	label_order=clf.classes_.tolist()
	claName=[]

	for k in label_order:
		for o in labels:
			if labels[o]== int(k):
				claName.append(o)

	print("Labeled order is ", label_order)
	print("class  name order is ", claName)
	#test the performance without learning unlabeled data
	test_pred,test_label=[],[]
	with open("csvfold/Test_" + str(i) + ".csv", "rb") as csvFile:
	    csvReader = csv.reader(csvFile, delimiter=' ')
	    for row in csvReader:
	      features = classify(net, [ row[0] ])
	      features2= classify(net2, [ row[0] ])

# the test classifier is not determined 	      
	      prediction = clf.predict(np.array(features[0]).reshape(1, lenofvector))
	      prediction2= clf2.predict(np.array(features2[0].reshape(1,lenofvector2)))
	      proba_list=clf.predict_proba(np.array(features[0]).reshape(1, lenofvector))
	      proba_list2=clf2.predict_proba(np.array(features2[0]).reshape(1, lenofvector2))
	      proba=proba_list[0][int(label_order.index(prediction))]
	      proba2=proba_list2[0][int(label_order.index(prediction2))]

#decide the terminal result 
	      if prediction == prediction2:
	      	prediction =prediction
	      else :
	      	if proba>=proba2:
	      		prediction= prediction
	      	else:
	      		prediction = prediction2

	      if prediction == row[1]:
	        correct += 1
	      else:
	        wrong += 1
	      total += 1
	  # paratmeter used for consturcting confusion matrix    
	      test_pred.append(prediction)
	      test_label.append(row[1])
	print("TOTAL: " + str(total))
	print("CORRECT: " + str(correct))
	print("WRONG: " + str(wrong))
	output = "the iteration round order is" + str(j)+"\n"
	output += "the accuracy ratio is " + str(float(correct) / float(total) * 100) + "\n\n"
	open("results.txt", "a").write(output + "\n")	
	# paint the confusion matrix graph 
	plot_save_graph(test_label,test_pred,0,claName)
	
	while True:
#		print( "the value of j is : " , str(j))
		addedsamplePerClass = [0 for k  in  range(21)]
		batch_sample, batch_y = unlabeled_sample[210*(j-1):210*j],unlabeled_y[210*(j-1):210*j]
# the order is the label order learnt in svm


# feature is the output vector of CNN
# use  svm to predict the unlabeled  and save the resutl into EL_sample and EL_y
# 
		for k in range (210):

			features = classify(net, [batch_sample[k]])
			features2 = classify(net2, [batch_sample[k]])
			prediction = clf.predict(np.array(features[0]).reshape(1, lenofvector))
			prediction2 = clf2.predict(np.array(features2[0]).reshape(1,lenofvector2))
			proba_list=clf.predict_proba(np.array(features[0]).reshape(1, lenofvector))
			proba_list2=clf2.predict_proba(np.array(features2[0]).reshape(1, lenofvector2))
			proba=proba_list[0][int(label_order.index(prediction))]
			proba2=proba_list2[0][int(label_order.index(prediction2))]
			# classPointer save the value of the lable of prediciton 
			
#			print("the accuracy of ", batch_sample[k], " is ", str(proba*100),"%")
#			
			if prediction[0]==prediction2[0] and (proba>=0.3 or proba2>=0.3):
				classPointer= int(prediction[0])
				if addedsamplePerClass[classPointer]<MaxNumPerClassPerIteration:
					addedsamplePerClass[classPointer]+=1
					EL_samples.append(batch_sample[k])
					EL_y.append(str(prediction[0]))
					addedsamplenum+=1
				else:
					lowConfidence_sample.append(batch_sample[k])
					lowConfidence_y.append(str(prediction[0]))
#			


#train the clf
			
		train_sample,train_X,train_X2, train_y=[],[],[],[]
		train_sample,train_y= labeled_sample+EL_samples, labeled_y+EL_y
		print("the len of train_sample is :", str(len(train_sample)), "the number of added samples is :",str(addedsamplenum))

		for k in range(len(train_sample)):
			features= classify(net,[train_sample[k]])
			train_X.append(features[0])
			features2= classify(net2,[train_sample[k]])
			train_X2.append(features2[0])
#		pdb.set_trace()	
		clf = calibration.CalibratedClassifierCV(svm.LinearSVC(C=100000))
		clf.fit(train_X,train_y)
		clf2 = calibration.CalibratedClassifierCV(svm.LinearSVC(C=100000))
		clf2.fit(train_X2,train_y)		

		correct = 0
		total = 0
		wrong = 0

		test_pred, test_label=[],[]
		with open("csvfold/Test_" + str(i) + ".csv", "rb") as csvFile:
		    csvReader = csv.reader(csvFile, delimiter=' ')
		    for row in csvReader:
		      features = classify(net, [ row[0] ])
		      features2= classify(net2, [ row[0] ])

	# the test classifier is not determined 	      
		      prediction = clf.predict(np.array(features[0]).reshape(1, lenofvector))
		      prediction2= clf2.predict(np.array(features2[0].reshape(1,lenofvector2)))
		      proba_list=clf.predict_proba(np.array(features[0]).reshape(1, lenofvector))
		      proba_list2=clf2.predict_proba(np.array(features2[0]).reshape(1, lenofvector2))
		      proba=proba_list[0][int(label_order.index(prediction))]
		      proba2=proba_list2[0][int(label_order.index(prediction2))]

	#decide the terminal result 
		      if prediction == prediction2:
		      	prediction =prediction
		      else :
		      	if proba>=proba2:
		      		prediction= prediction
		      	else:
		      		prediction = prediction2

		      if prediction == row[1]:
		        correct += 1
		      else:
		        wrong += 1
		      total += 1
		      test_pred.append(prediction)
		      test_label.append(row[1])
		print("TOTAL: " + str(total))
		print("CORRECT: " + str(correct))
		print("WRONG: " + str(wrong))
		output = "the iteration round order is" + str(j)+"\n"
		output += "the accuracy ratio is " + str(float(correct) / float(total) * 100) + "\n\n"
		open("results.txt", "a").write(output + "\n")	

		plot_save_graph(test_label,test_pred,j,claName)

		j+=1
		if j>8:
			break      

	print ("train the low confidence samples")

	#count save the  last time value of addedsamplenum
	count=0
# iteration learning the lowconfidence samples
	while count!=addedsamplenum:
		j=j+1
		lowConfidenceMid_sample=[]		
		count=addedsamplenum

		for k in range (len(lowConfidence_sample)):
			sample= lowConfidence_sample[k]
			features = classify(net, [sample])
			features2 = classify(net2, [sample])
			prediction = clf.predict(np.array(features[0]).reshape(1, lenofvector))
			prediction2 = clf2.predict(np.array(features2[0]).reshape(1,lenofvector2))
			proba_list=clf.predict_proba(np.array(features[0]).reshape(1, lenofvector))
			proba_list2=clf2.predict_proba(np.array(features2[0]).reshape(1, lenofvector2))
			proba=proba_list[0][int(label_order.index(prediction))]
			proba2=proba_list2[0][int(label_order.index(prediction2))]

			# classPointer save the value of the lable of prediciton 
#			print("the accuracy of ", batch_sample[k], " is ", str(proba*100),"%")
			if prediction[0]==prediction2[0] and (proba>=0.3 or proba2>=0.3): 
				EL_samples.append(sample)
				EL_y.append(str(prediction[0]))
				addedsamplenum+=1
			else:
				lowConfidenceMid_sample.append(sample)

		train_sample,train_X,train_y,train_X2=[],[],[],[]
		train_sample,train_y= labeled_sample+EL_samples, labeled_y+EL_y
		print("the len of train_sample is :", str(len(train_sample)), "the number of added samples is :",str(addedsamplenum))

		for k in range(len(train_sample)):
			features= classify(net,[train_sample[k]])
			train_X.append(features[0])
			features2= classify(net2,[train_sample[k]])
			train_X2.append(features2[0])
		#		pdb.set_trace()	
		clf = calibration.CalibratedClassifierCV(svm.LinearSVC(C=100000))
		clf.fit(train_X,train_y)
		clf2 = calibration.CalibratedClassifierCV(svm.LinearSVC(C=100000))
		clf2.fit(train_X2,train_y)	


		correct = 0
		total = 0
		wrong = 0

		test_label,test_pred=[],[]
		with open("csvfold/Test_" + str(i) + ".csv", "rb") as csvFile:
		    csvReader = csv.reader(csvFile, delimiter=' ')
		    for row in csvReader:
		      features = classify(net, [ row[0] ])
		      features2= classify(net2, [ row[0] ])

	# the test classifier is not determined 	      
		      prediction = clf.predict(np.array(features[0]).reshape(1, lenofvector))
		      prediction2= clf2.predict(np.array(features2[0].reshape(1,lenofvector2)))
		      proba_list=clf.predict_proba(np.array(features[0]).reshape(1, lenofvector))
		      proba_list2=clf2.predict_proba(np.array(features2[0]).reshape(1, lenofvector2))
		      proba=proba_list[0][int(label_order.index(prediction))]
		      proba2=proba_list2[0][int(label_order.index(prediction2))]

	#decide the terminal result 
		      if prediction == prediction2:
		      	prediction =prediction
		      else :
		      	if proba>=proba2:
		      		prediction= prediction
		      	else:
		      		prediction = prediction2

		      if prediction == row[1]:
		        correct += 1
		      else:
		        wrong += 1
		      total += 1
		      test_pred.append(prediction)
		      test_label.append(row[1])
		print("TOTAL: " + str(total))
		print("CORRECT: " + str(correct))
		print("WRONG: " + str(wrong))
		output = "the iteration round order is" + str(j)+"\n"
		output += "the accuracy ratio is " + str(float(correct) / float(total) * 100) + "\n\n"
		open("results.txt", "a").write(output + "\n")	
		plot_save_graph(test_label,test_pred,j,claName)
		lowConfidence_sample=[]
		lowConfidence_sample=lowConfidenceMid_sample







def go (i):
	global SVM_deployPath, caffeModelPath, SVM_deployPath2, caffeModelPath2
	open("results.txt","a").write("the trainmeod is CaffeNet")
	SVM_deployPath = "deploy_svm_caffenet.prototxt"
	caffeModelPath = "/home/hpc-126/caffe-master/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
	SVM_deployPath2 = "/home/hpc-126/caffe-master/models/VGG_F/VGG_CNN_F_deploy_svm.prototxt"
	caffeModelPath2 = "/home/hpc-126/caffe-master/models/VGG_F/VGG_CNN_F.caffemodel"
	svmTrain(i)

def main(argv):
	if len(argv)<2:
		print("Usage: python test.py <command>")
		print("Command is :")
		print(" - clean")
		print(" - generate")
		print(" - go")
		return;

	if argv[1] == "clean":
		remove_dir("DataSet_JPG")

	elif argv[1]=="generate":
		images = convert_images("DataSet")
		convert_data_to_csv(images)

	elif argv[1]== "go":
		if len(argv)==3:
			go(int(argv[2]))
		else:

			for i in range (11):
				go (i)

if __name__=="__main__":
	main(sys.argv)

		





