import sys
import os
import numpy as np 
import pandas as pd
from keras.utils import np_utils

df = pd.read_csv('../fer2013/fer2013/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')
#print(df.head())

x_train,x_test,y_train,y_test = [],[],[],[]

#row[0]:emotion,row[1]:Usage,row[2]:pixels(48*48)
for index,row in df.iterrows():
	intensity_val=row[2].split(" ")
	try:
		if 'Training' in row[1]:
			x_train.append(np.array(intensity_val,'float32'))
			y_train.append(row['emotion'])
		elif 'PrivateTest' in row[1]:
			x_test.append(np.array(intensity_val,'float32'))
			y_test.append(row['emotion'])
	except:
		print(f"error occured at index:{index} and row:{row}")

#print("x_train",x_train[0:2])
#print("x_test",x_test[0:2])	
#print("y_train",y_train[0:2])
#print("y_test",y_test[0:2])

x_train=np.array(x_train,'float32')
x_test=np.array(x_test,'float32')
y_train=np.array(y_train,'float32')
y_test=np.array(y_test,'float32')

#normalizing data b/n 0 and 1
x_train-=np.mean(x_train,axis=0)
x_train/=np.std(x_train,axis=0)

x_test-=np.mean(x_test,axis=0)
x_test/=np.std(x_test,axis=0)

#print(x_train.shape)
#x_train.shape[0] - num_of_rows
num_of_filters=32 #initial layer
num_of_labels=7
batch_size=64 #divison of input size into batches
epochs=60 #num of iterations
width,height=48,48

#print(x_train.shape[1:])
#print(x_train.shape)
x_train=x_train.reshape(x_train.shape[0],width,height,1)
#print(x_train.shape[1:])
#print(x_train.shape)
x_test=x_test.reshape(x_test.shape[0],width,height,1)

y_train=np_utils.to_categorical(y_train,num_of_labels)
y_test=np_utils.to_categorical(y_test,num_of_labels)






	


