"""
    Coding by Nguyen Duc Huy
    Regression Method for Neural Network
"""

# import library
import Neural_Network_Model
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# import PlotRealtime as pltrealtime

#load data set
data = pd.read_csv('energy_efficiency_data.csv')
t_heating = data.iloc[:,8].values
t_heating = np.array(t_heating).reshape(-1,1)
input_data = data.drop(columns= ['Heating Load','Cooling Load'])
input_data = pd.get_dummies(input_data,columns=['Orientation','Glazing Area Distribution'])
input_data = input_data.values
#len_row_input_data = len(input_data)
#len_column_input_data = len(input_data[0])
#input_data = np.array(input_data).reshape(len_row_input_data,len_column_input_data)


#NORMALIZE
input_data[:,0:6] = input_data[:,0:6]/input_data[:,0:6].max(axis=0) # normolize for 6 input no one hot vector

maxHeatingLoad= t_heating.max(axis=0)
t_heating = t_heating/maxHeatingLoad

#Calculate Variance
variance = []
for i in range(6):
    std = np.std(input_data[:,i])
    variance.append(std)

#FEATURE SELECTION WHICH STANDARD DEVIATION
#REMOVE SOME OF FEATURES WITH LOW STANDARD DEVIATION
NumberOfRemovedFeaturesSmallestVariance = 0
NumberOfRemovedFeaturesLargestVariance = 1


for i in range(NumberOfRemovedFeaturesSmallestVariance):
    removedTempFeature = variance.index(min(variance))
    variance.remove(min(variance))
    input_data = np.delete(input_data, removedTempFeature, 1)
    print("RemovedFeatures: " + str(removedTempFeature) + "th" )

for i in range(NumberOfRemovedFeaturesLargestVariance):
    removedTempFeature = variance.index(max(variance))
    variance.remove(max(variance))
    input_data = np.delete(input_data,removedTempFeature,1)
    print("RemovedFeatures: " + str(removedTempFeature) + "th")

print(input_data)
x_train, x_test, t_train, t_test = train_test_split(input_data,t_heating,test_size=0.25, shuffle=True)

# built model neuron network
dim_x = len(x_train[0])
neuron_shape = [dim_x,10,10,1]
#activation_function = ['tanh','tanh','linear']
#activation_function = ['relu','relu','linear'] # RELU is wrong so in this my code shuoldn't use RELU
activation_function = ['sigmoid','sigmoid','linear']
learning_rate = 0.002

model = Neural_Network_Model.NeuronNetwork4Regression(neuron_shape,activation_function,learning_rate,True)

#  epoch = 1000
epoch = 10000
batch_size = 16

model.init_weight()
print('=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=')
print(model.get_weight())
E_rms = model.E_rms(x_train,t_train)


#Print Figure for Training Error every EPOCHS (In this exercise, Epochs from 1 to 1000)
#Print Results of Traning LOSS
flagcounter = 0
ErrorRMSTraining4Plot = []
ErrorRMSTesting4Plot = []
ErrorSumSquareTraining4Plot = []
ErrorSumSquareTesting4Plot = []
line1 = []
for i in range(epoch):
    model.gradient_descent(x_train,t_train,batch_size)
    E_rms_training = model.E_rms(x_train,t_train)
    ErrorRMSTraining4Plot.append(E_rms_training)
    E_rms_testing = model.E_rms(x_test,t_test)
    ErrorRMSTesting4Plot.append(E_rms_testing)
    E_square_error_training = (model.E_Sum_Square_Error(x_train,t_train))/(len(x_train)/batch_size)
    E_square_error_testing = (model.E_Sum_Square_Error(x_test,t_test))/(len(x_test)/batch_size)
    ErrorSumSquareTraining4Plot.append(E_square_error_training)
    ErrorSumSquareTesting4Plot.append(E_square_error_testing)
    #print(E_rms)
    flagcounter = flagcounter + 1
    if flagcounter % 1000 == 0:
        print('Epoch = {}, the sum-of-squares error: {} and root-mean-square error (RMS): {}'.format(flagcounter, float(E_square_error_training), float(E_rms_training)))
        #plt.scatter(np.array(ErrorTraining4Plot).reshape(-1,1), '-g', label = 'Training Curver Error')
        #plt.scatter(np.array(ErrorTesting4Plot).reshape(-1,1), '-r', label = 'Testing Curve Error')
        #plt.pause(0.05)
        #plt(epoch, E_rms)
    # PLOTTING ERROR TRAINING AND TESTING BASED ON EPOCHS (IN THIS EXERCISE, I USED FROM 1 TO 1000)
    #line1 = pltrealtime.live_plotter(np.array(ErrorTraining4Plot).reshape(-1,1),flagcounter,line1)

#PLOTTING ERROR TRAINING AND TESTING BASED ON EPOCHS (IN THIS EXERCISE, I USED FROM 1 TO 1000)
plt.figure(figsize = (8,4))
plt.plot(np.array(ErrorSumSquareTraining4Plot).reshape(-1,1),'-g', label = 'Training Curve Error',linewidth=2.0)
#plt.plot(np.array(ErrorTesting4Plot).reshape(-1,1),'-r', label = 'Testing Curve Error')
plt.title("Training Curve")
plt.xlabel("Number of Epochs ")
plt.ylabel("Sum-Of-Squares Error")
plt.legend(loc = 'best')
plt.show()

'''
for i in range(epoch):
    model.gradient_descent(x_train,t_train,batch_size)
    ErrorTraining = model.E_rms(x_train,t_train)
    ErrorTesting = model.E_rms(x_test,t_test)
    flagcounter += 1
    if flagcounter%50 == 0:
        ErrorTraining4Plot.append(ErrorTraining)
        ErrorTesting4Plot.append(ErrorTesting)
    plt.plot(epoch,ErrorTraining4Plot,'-g',label = 'Training Curver Error')
    plt.plot(epoch,ErrorTesting4Plot,'-r',label = 'Testing Curve Error')
    plt.show()
'''

print("=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=")
print('=:=:=:=:=:=:=:=:=:The last result of Error Training:=:=:=:=:=:=:=:=')
E_rms_4training = model.E_rms(x_train,t_train)
E_rms_4testing = model.E_rms(x_test, t_test)
print('Error RMS Training Result: {} and Testing Result: {}'.format(E_rms_4training,E_rms_4testing))
print("=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=")
#E_Square_Error_4Training = model.E_Sum_Square_Error(x_train,t_train)
#E_Square_Error_4Testing = model.E_Sum_Square_Error(x_test,t_test)
#print('Sum-Of-Squares Error Training Result: {} and Testing Result: {}'.format(E_Square_Error_4Training,E_Square_Error_4Testing))
print('Sum-Of-Squares Error Training Result: {} and Testing Result: {}'.format(ErrorSumSquareTraining4Plot[-1],ErrorSumSquareTesting4Plot[-1]))
print("=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=")
print("=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=")



x_train_shuffle, x_test_shuffle, t_train_shuffle, t_test_shuffle = train_test_split(input_data,t_heating,test_size=0.25, shuffle=True)
x_train_no_shuffle, x_test_no_shuffle, t_train_no_shuffle, t_test_no_shuffle = train_test_split(input_data,t_heating,test_size=0.25, shuffle=False)

y_predict_training = model.y_predict(x_train_no_shuffle)
plt.figure(figsize = (8,4))
plt.plot(y_predict_training, 'r-', label = 'prediction',linewidth = 2)
plt.plot(t_train_no_shuffle, 'g-', label=' label',linewidth = 2)
plt.title("Prediction for Training Data with No Shuffle")
plt.xlabel("#th Case")
plt.ylabel("Heating Load")
plt.legend(loc='best') # tạo nhãn dán label chú thích các tên của các đường LINE được vẽ
plt.show()

plt.figure(figsize = (4,3))
y_predict_testing = model.y_predict(x_test_no_shuffle)
plt.plot(y_predict_testing, 'r-', label = 'prediction',linewidth = 2)
plt.plot(t_test_no_shuffle, 'g-', label=' label',linewidth = 2)
plt.title("Prediction for Testing Data with No Shuffle")
plt.xlabel("#th Case ")
plt.ylabel("Heating Load")
plt.legend(loc='best') # tạo nhãn dán label chú thích các tên của các đường LINE được vẽ
plt.show()

plt.figure(figsize = (8,4))
y_predict_testing_shuffle = model.y_predict(x_train_shuffle)
plt.plot(y_predict_testing_shuffle, 'r-', label = 'prediction',linewidth = 2)
plt.plot(t_train_shuffle, 'g-', label=' label',linewidth = 2)
plt.title("Prediction for Training Data with Shuffle")
plt.xlabel("#th Case ")
plt.ylabel("Heating Load")
plt.legend(loc='best') # tạo nhãn dán label chú thích các tên của các đường LINE được vẽ
plt.show()

plt.figure(figsize = (4,3))
y_predict_testing_shuffle = model.y_predict(x_test_shuffle)
plt.plot(y_predict_testing_shuffle, 'r-', label = 'prediction',linewidth = 2)
plt.plot(t_test_shuffle, 'g-', label=' label',linewidth = 2)
plt.title("Prediction for Testing Data with Shuffle")
plt.xlabel("#th Case ")
plt.ylabel("Heating Load")
plt.legend(loc='best') # tạo nhãn dán label chú thích các tên của các đường LINE được vẽ
plt.show()
