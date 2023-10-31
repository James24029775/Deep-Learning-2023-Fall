"""
    Coding by Nguyen Duc Huy
    Classification Method for Neural Network
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import Neural_Network_Model
import csv
from mpl_toolkits.mplot3d import Axes3D

#Load Data Set
import os
print(os.getcwd())
data = pd.read_csv('ionosphere_data.csv')
input_data = data.iloc[:,0:34].values


target = data.iloc[:,34].values
target_onehot = []
for i in range(len(target)):
    if (target[i] == 'g'):
        target_onehot.append([1,0])
    else:
        target_onehot.append([0,1])
target_onehot = np.reshape(target_onehot,(len(target),2))
'''
nnFeature_Total = 34
target_onehot = []
#Assign OneHot Vector for Targert "Good-G" and "Bad-B"
#with open('ionosphere_data.csv') as csv_file: Anh Giang Code Đoạn này
with open('ionosphere_csv.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    #temp = 0
    for row in csv_reader:
        for j in range(nnFeature_Total):
            target_onehot.append(float(row[j]))
        if (row[nnFeature_Total] == 'g'):
            target_onehot.append(1)
        else:
            target_onehot.append(0)
'''

#Normalization

#cc = input_data[:,0:34].max(axis=0)
# Normalizing for 33 inputs from 1 to 33 no one hot vector,
# And feature a01 is Onehot vector so It doesn't need to be normalized

#Print input_data for checking
print(input_data)
x_train, x_test, t_train, t_test = train_test_split(input_data,target_onehot,test_size=0.2, shuffle=False)

#Build Neural Network Model for Classification based on the NNModel Designed by myself
dim_x = np.array(x_train).shape[1]
print(dim_x)
neural_shape = [dim_x,128,64,32,3,2]
activation_function = ['sigmoid','sigmoid','sigmoid','sigmoid','softmax']
learning_rate = 0.002


model = Neural_Network_Model.NeuronNetwork4Classification(neural_shape,activation_function,learning_rate,False,1)


epoch = 1000
batch_size = 32

model.init_weight()
print('=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=')
print(model.get_weight())
E_rms = model.E_rms_cross_entropy(x_train,t_train)
print(E_rms)
a = 0
E_Rate_train_4_plot = []
E_Rate_test_4_plot = []
E_Cross_Entropy_4_plot = []
meanH1Good =[]
meanH1Bad =[]
meanH1Good999 =[]
meanH1Bad999 =[]
for i in range(epoch):
    model.gradient_descent(x_train,t_train,batch_size)
    E_rms_cross_entropyy = model.E_rms_cross_entropy(x_train, t_train)
    E_Rate_train = model.E_Rate(x_train,t_train)
    E_Rate_train_4_plot.append(E_Rate_train)
    E_Rate_test = model.E_Rate(x_test, t_test)
    E_Rate_test_4_plot.append(E_Rate_test)
    E_Cross_Entropy_4_plot.append(E_rms_cross_entropyy)
    #print(E_rms)
    a = a + 1
    if a % 200 == 0:
        print('Epoch = {}, Loss of Training: {}'.format(a, E_rms_cross_entropyy))
    #plt(epoch, E_rms)

    #PLOTTING for
    target_onehot_new = []
    _temp = len(target_onehot)
    for i in range(_temp):
        if (target[i]=='g'):
            target_onehot_new.append(1)
        else:
            target_onehot_new.append(0)
    #target_onehot_new = np.reshape(-1,1)

    # PLOTTING

    if (a == 5):
        _input = input_data
        _rawOutput = target_onehot_new
        model.feed_forward(_input)
        for k in range(len(x_train)):
            if(_rawOutput[k]==1):
                meanH1Good.append(model.A[-2][k]) #thực chất ra để như kiểu coding này cũng được: meanH1Good.append(model.A[-2][k])
            else:
                A = model.A[-2]
                meanH1Bad.append(model.A[-2][k])

    if (a == 390):
        _input = input_data
        _rawOutput = target_onehot_new
        predictValue = model.feed_forward(_input)
        for k in range(len(x_train)):
            if (_rawOutput[k]==1):
                meanH1Good999.append(model.A[-2][k])
            else:
                meanH1Bad999.append(model.A[-2][k])



#y_predict = model.y_predict(input_data)
#plt.plot(y_predict, 'r-', label = 'prediction')
#plt.plot(t_heating, 'g-', label=' label')
#plt.legend(loc='best') # tạo nhãn dán label chú thích các tên của các đường LINE được vẽ

print("=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=")
print(':=:=:=:=:=:=:=:=The last result of Error Training:=:=:=:=:=:=:=:=')
E_rms = model.E_rms_cross_entropy(x_train,t_train)
print('Error Training Result: {}'.format(E_rms))
print("=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=")
print(':=:=:=:=:=:=:=:=The last result of Error Testing:=:=:=:=:=:=:=:=')
E_test = model.E_rms_cross_entropy(x_test, t_test)
print('Error Testing Result: {}'.format(E_test))
print("=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=")
print(':=:=:=:=:=:=:=:=The last result of Error Rate Training and Testing:=:=:=:=:=:=:=:=')
E_Rate_train = model.E_Rate(x_train, t_train)
E_Rate_test = model.E_Rate(x_test, t_test)
print('Error Training Rate Result: {} % and Error Testing Rate Result: {} %'.format(E_Rate_train,E_Rate_test))
print("=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=")

epoch4plot = range(0,epoch,1)
plt.plot(epoch4plot,E_Rate_train_4_plot, 'r-', label = 'prediction')
plt.title("Error Rate Training Phase")
plt.xlabel("#th Epoches ")
plt.ylabel("Error Rate")
plt.legend(loc='best') # tạo nhãn dán label chú thích các tên của các đường LINE được vẽ
plt.show()

plt.plot(epoch4plot,E_Cross_Entropy_4_plot, 'r-', label = 'prediction')
plt.title("Cross Entropy Training Phase")
plt.xlabel("#th Epoches ")
plt.ylabel("Cross Entropy Error")
plt.legend(loc='best') # tạo nhãn dán label chú thích các tên của các đường LINE được vẽ
plt.show()


plt.plot(epoch4plot,E_Rate_train_4_plot, 'r-', label = 'training',linewidth = 3)
plt.plot(epoch4plot,E_Rate_test_4_plot, 'b-', label = 'testing',linewidth = 3)
plt.title("Error Rate Training and Testing Phase")
plt.xlabel("#th Epoches ")
plt.ylabel("Error Rate")
plt.legend(loc='best') # tạo nhãn dán label chú thích các tên của các đường LINE được vẽ
plt.show()


meanH1Good999 = np.array(meanH1Good999)
meanH1Bad999 = np.array(meanH1Bad999)
meanH1Good = np.array(meanH1Good)
meanH1Bad = np.array(meanH1Bad)

plt.figure(figsize = (4, 4))
plt.subplot(121)
plt.plot(meanH1Good[:, 0], meanH1Good[:, 1], "ro", label = "Good")
plt.plot(meanH1Bad[:, 0], meanH1Bad[:, 1], "bo", label = "Bad")
plt.title("2D feature 5th epoch")
plt.subplot(122)
plt.plot(meanH1Good999[:, 0], meanH1Good999[:, 1], "ro", label = "Good")
plt.plot(meanH1Bad999[:, 0], meanH1Bad999[:, 1], "bo", label = "Bad")
plt.title("2D feature 999th epoch")
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()

ax = plt.subplot(121, projection = '3d')
ax.scatter(meanH1Good[:, 0], meanH1Good[:, 1], meanH1Good[:, 2], marker = 'o', color = "green", label = "Good", alpha = 1.0)
ax.scatter(meanH1Bad[:, 0], meanH1Bad[:, 1], meanH1Bad[:, 2], marker = 'o', color = "red", label = "Bad", alpha = 1.0)
plt.title("3D feature 5th epoch")
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
ax = plt.subplot(122, projection = '3d')
ax.scatter(meanH1Good999[:, 0], meanH1Good999[:, 1], meanH1Good999[:, 2], marker = 'o', color = "green", alpha = 1.0, label = "Good")
ax.scatter(meanH1Bad999[:, 0], meanH1Bad999[:, 1], meanH1Bad999[:, 2], marker = 'o', color = "red", alpha = 1.0, label = "Bad")
plt.title("3D feature 999th epoch")
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()








