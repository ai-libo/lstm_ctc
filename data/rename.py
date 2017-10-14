#-*- coding:utf8 -*-
import os
import natsort

dirnames = os.listdir("train")
dirnames = natsort.natsorted(dirnames)
i=0
for d in dirnames:
	files = os.listdir("train/"+d)
	files = natsort.natsorted(files)
	# for f in files:
	# 	if f[-3:] == "jpg":
	# 		os.rename("train/"+d+'/'+f,"train/"+d+'/'+str(i)+'.jpg')
	# 		i = i+1
	with open("train/"+d+'/'+'无标题文档','r') as f:
		lines = f.readlines()
		for k in range(len(lines)):
			f = files[k]
			os.rename("train/"+d+'/'+f,"train/"+d+'/'+str(i)+'.jpg')
			g = open("train/"+d+'/'+str(i)+'.txt','w') 
			g.write(lines[k])
			g.close()
			i = i+1

