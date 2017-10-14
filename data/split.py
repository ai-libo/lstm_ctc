#-*-coding:utf8 -*-

import cv2
import numpy as np 
import pdb
import os

def split(file):
	im = cv2.imread(file)
	im = im[:,:,0]
	shape = im.shape#(4160,3120)
	j = []
	k = []
	for i in range(shape[0]):
		# if np.sum(im[i,:]==255)/shape[0] > 0.999999:
		if np.where(im[i,:] == 255)[0].shape[0] / shape[1]>0.999:
			j.append(i)
		else:
			if j != []:
				s = len(j)//2
				k.append(j[s])
			j = []
	for i in range(len(k)-1):
		crop = im[k[i]:k[i+1],:]
		if not os.path.exists(file[:-4]):
			os.mkdir(file[:-4])
		path = file[:-4]+'/'+str(i)+'.jpg'
		cv2.imwrite(path,crop)
files = os.listdir('binar2')
for file in files:
	split('binar2/'+file)

import os 
import cv2
l = [5]
for j in range(1):
	k = l[j]
	files = os.listdir("binar2/x"+str(k)+".bin")

	for file in files:
		path = "binar2/x"+str(k)+".bin/"+file
		im = cv2.imread(path)
		shape = im.shape
		if shape[0] <50:
			os.remove(path)












