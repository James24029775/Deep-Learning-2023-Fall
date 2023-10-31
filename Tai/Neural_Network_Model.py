"""
    Coding by Nguyen Duc Huy
    Neural Network Model Using For Gradient Descent Method
"""

import numpy as np

import math  # Add this import

# define activation function and derivative of activation function

# your_array = your_array.astype(float)
# output = np.exp(your_array)


def sigmoid(x):
    x=np.array(x,dtype=np.float64)
    return 1 / (1 + np.exp(-x))


def softmax(x):
	returnValue = np.exp(x - np.max(x, axis=0))
	return returnValue/np.sum(returnValue, axis=0)



def derivative_sigmoid(a):
    return a * (1 - a)


def relu(x):
    return np.maximum(0., x)


def derivative_relu(x):
    return np.maximum(np.minimum(1, np.round(x + 0.5)), 0)


def softmax(x):
    tmp = np.exp(x - np.max(x))
    return tmp/tmp.sum(axis = 1, keepdims = True)


def tanh(x):
    return np.tanh(x)


def derivative_tanh(x):
    return 1 - np.tanh(x) ** 2


def linear(x):
    return x


def derivative_linear(x):
    return 1.





def act_function(x, action):
    if action == 'sigmoid':
        return sigmoid(x)
    elif action == 'relu':
        return relu(x)
    elif action == 'softmax':
        return softmax(x)
    elif action == 'tanh':
        return tanh(x)
    elif action == 'linear':
        return linear(x)

def derivative_function(x, action):
    if action == 'sigmoid':
        return derivative_sigmoid(x)
    elif action == 'relu':
        return derivative_relu(x)
    elif action == 'softmax':
        return 1
    elif action == 'tanh':
        return derivative_tanh(x)
    elif action == 'linear':
        return derivative_linear(x)



#=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
#=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
#Build the Neural Network 4 Regression
#=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
#=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
class NeuronNetwork4Regression:
    def __init__(self, neuron_shape, activation_function, learning_rate, flagcounter = True, seedd = 0):
        """
            this function to init Neuron network
            W: list include of weight matrix of each layer
            number_layer: number of layer, not include of input matrix x
            b: list include of bias vector of each layer
            A: list include of output matrix of each layer, A(l) = f(z)    z = W_T*x + b   A(l): (n x d(l))
            A include input layer in the first element list
            dJ_W: list include of dJ/dW matrix of each layer
            dJ_b: list include of dJ_db vector of each layer
            dJ_A: list include of dJ_dA matrix of each layer
        :param neuron_shape:  row vector include the number of node in each layer
        :param activation_function: activation function in each layer
        :param learning_rate: learning rate in GD
        """
        self.neuron_shape = neuron_shape
        self.number_layer = len(neuron_shape) - 1
        self.activation_function = activation_function
        self.LR = learning_rate
        self.W = []
        self.b = []
        self.A = []
        # self.e = []  # e = dZ = [W(l+1)*e(l+1)] (*) df(z)
        self.dJ_W = []  # dJ/dW
        self.dJ_b = []  # dJ/db
        self.dJ_A = []  # dJ/dA
        self.gama_momentum = 0.9
        self.flagcounter = flagcounter
        self.seedd = seedd

    def init_weight(self):
        """
        This function to init weight and bias for each layer
        :return: NULL
        """
        for i in range(self.number_layer,):
            if not self.flagcounter:  # đây có nghĩa là == False
                np.random.seed(self.seedd)
            weight = np.random.randn(self.neuron_shape[i], self.neuron_shape[i + 1]) * np.sqrt(2 / self.neuron_shape[i])
            bias = np.zeros([self.neuron_shape[i + 1], 1])
            self.W.append(weight)
            self.b.append(bias)
        # self.momentum_chuanhan = np.zeros_like(self.W)
        # self.momentumbias_chuanhan = np.zeros_like(self.b)
        self.momentum_chuanhan = [np.zeros_like(w) for w in self.W]
        self.momentumbias_chuanhan = [np.zeros_like(bi) for bi in self.b]

    def feed_forward(self, x):
        """
        this function to calculate the list of output matrix in each layer, include input matrix x
        :param x: input matrix x
        :return: NULL
        """
        # reset vector A
        self.A = []
        # x input: n*m
        self.A.append(x)
        for i in range(self.number_layer):
            input_layer = self.A[-1]   #ở ma trận A, A[-1] tức là lấy hàng cuối cùng
            z = np.dot(input_layer, self.W[i]) + self.b[i].T
            self.A.append(act_function(z, self.activation_function[i]))

    def back_propagation(self, x, t):
        """
        this function to calculate derivative of weight and bias of each layer
        :param x: input matrix x (nxm)
        :param t: output matrix t  (nx1)
        :return: NULL
        """
        # reset vector
        self.dJ_b = []  #Đạo hàm theo Bias
        self.dJ_W = []  #Đạo hàm theo Weight
        self.dJ_A = []  #Đạo hàm theo A
        # first: calculate dJ_dA(L)=2*(A(L)-t)   L = number_layer ; A(0) = x (nxm)
        t = np.array(t).reshape(-1, 1)
        AL = self.A[-1]
        dJ_AL = 2 * (AL - t)
        self.dJ_A.append(dJ_AL)

        # calculate dJ_W(l),dJ_b(l),dJ_A(L-1)
        for i in reversed(range(0,self.number_layer)):
            # define for step 0
            Al_subtract_1 = self.A[i]  # dim of A= number_layer + 1, but len(i) = number_layer  => A(l-1) = A[i]
            Al = self.A[i + 1]
            dJ_Al = self.dJ_A[-1]
            dAl_dZ = derivative_function(Al, self.activation_function[i])
            Wl = self.W[i]

            # append in the list
            self.dJ_W.append(np.dot(Al_subtract_1.T, dJ_Al * dAl_dZ))
            self.dJ_b.append(np.sum(dJ_Al * dAl_dZ, 0).reshape(-1, 1))
            self.dJ_A.append(np.dot((dJ_Al * dAl_dZ), Wl.T))

        # reversed dJ_W and dJ_b
        self.dJ_W = self.dJ_W[::-1]
        self.dJ_b = self.dJ_b[::-1]

    def update_weight(self):
        """
        this function to optimize weight and bias
        :return: NULL
        """
        


        for i in range(self.number_layer):
            #self.W[i] = self.W[i] - self.LR * self.dJ_W[i]
            #self.b[i] = self.b[i] - self.LR * self.dJ_b[i]
            self.W[i] = self.W[i] - self.LR*self.dJ_W[i] - self.LR*self.gama_momentum*self.momentum_chuanhan[i]
            self.b[i] = self.b[i] - self.LR*self.dJ_b[i] - self.LR*self.gama_momentum*self.momentumbias_chuanhan[i]
        self.momentum_chuanhan = self.dJ_W
        self.momentumbias_chuanhan = self.dJ_b

    def gradient_descent(self,x,t,batch_size):
        """
        this function to update weight and bias in one epoch
        :param x: input matrix x
        :param t: output matrix t
        :param batch_size: batch size, the data in final batch maybe < batch size
        :return: NULL
        """
        data_size = len(t)
        iterations = 0
        flag_batch_size = True
        if (data_size % batch_size) == 0:
            iterations = data_size/batch_size
        else:
            iterations = data_size//batch_size + 1
            flag_batch_size = False

        if flag_batch_size == True:
            for i in range(int(iterations)):
                in_data = x[i * batch_size:(i + 1) * batch_size]
                out_data = t[i * batch_size:(i + 1) * batch_size]

                self.feed_forward(in_data)
                self.back_propagation(in_data, out_data)
                self.update_weight()
        else:
            for i in range(int(iterations-1)):
                in_data = x[i * batch_size:(i + 1) * batch_size]
                out_data = t[i * batch_size:(i + 1) * batch_size]

                self.feed_forward(in_data)
                self.back_propagation(in_data, out_data)
                self.update_weight()
            in_data = x[int(iterations-1) * batch_size:]
            out_data = t[int(iterations-1) * batch_size:]
            self.feed_forward(in_data)
            self.back_propagation(in_data, out_data)
            self.update_weight()

    def y_predict(self,x):
        """
        this function to calculate y_predict
        :param x: input data matrix nxm
        :return: y_predict nx1
        """
        out_layer = np.array(x)
        for i in range(self.number_layer):
            z = np.dot(out_layer, self.W[i]) + self.b[i].T
            out_layer = act_function(z, self.activation_function[i])
        return out_layer.reshape(-1,1)

    def E_rms(self, x, t):
        """
        this function to calculate E_RMS
        :param x: input vector
        :param t: output vector
        :return: E_rms
        """
        y_predict = self.y_predict(x)
        N = len(y_predict)
        E_rms = np.sqrt(np.dot((t - y_predict).T,(t-y_predict))/N)
        return E_rms

    def E_Sum_Square_Error(self,x,t):
        y_predict = self.y_predict(x)
        N = len(y_predict)
        E_Sum_Square_Error = (np.dot((t-y_predict).T,(t-y_predict)))
        return E_Sum_Square_Error

    def get_weight(self):
        return self.W
    def get_bias(self):
        return self.b


#=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
#=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
#Build the Neural Network 4 Classification
#=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
#=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=
class NeuronNetwork4Classification:
    def __init__(self, neuron_shape, activation_function, learning_rate, flagcounter = True, seedd = 0):
        """
            this function to init Neuron network
            W: list include of weight matrix of each layer
            number_layer: number of layer, not include of input matrix x
            b: list include of bias vector of each layer
            A: list include of output matrix of each layer, A(l) = f(z)    z = W_T*x + b   A(l): (n x d(l))
            A include input layer in the first element list
            dJ_W: list include of dJ/dW matrix of each layer
            dJ_b: list include of dJ_db vector of each layer
            dJ_A: list include of dJ_dA matrix of each layer
        :param neuron_shape:  row vector include the number of node in each layer
        :param activation_function: activation function in each layer
        :param learning_rate: learning rate in GD
        """
        self.neuron_shape = neuron_shape
        self.number_layer = len(neuron_shape) - 1
        self.activation_function = activation_function
        self.LR = learning_rate
        self.W = []
        self.b = []
        self.A = []
        # self.e = []  # e = dZ = [W(l+1)*e(l+1)] (*) df(z)
        self.dJ_W = []  # dJ/dW
        self.dJ_b = []  # dJ/db
        self.dJ_A = []  # dJ/dA
        self.gama_momentum = 0.9
        self.flagcounter = flagcounter
        self.seedd = seedd

    def init_weight(self):
        """
        This function to init weight and bias for each layer
        :return: NULL
        """
        if not self.flagcounter:  # đây có nghĩa là == False
            np.random.seed(self.seedd)

        for i in range(self.number_layer):
            weight = np.random.randn(self.neuron_shape[i], self.neuron_shape[i+1])
            #weight = np.random.randn(self.neuron_shape[i], self.neuron_shape[i + 1]) * np.sqrt(2 / self.neuron_shape[i])
            bias = np.zeros([self.neuron_shape[i + 1], 1])
            self.W.append(weight)
            self.b.append(bias)
        self.momentum_chuanhan = [np.zeros_like(w) for w in self.W]
        self.momentumbias_chuanhan = [np.zeros_like(bi) for bi in self.b]
        # self.momentum_chuanhan = np.zeros_like(self.W)
        # self.momentumbias_chuanhan = np.zeros_like(self.b)
    def feed_forward(self, x):
        """
        this function to calculate the list of output matrix in each layer, include input matrix x
        :param x: input matrix x
        :return: NULL
        """
        # reset vector A
        self.A = []
        # x input: n*m
        self.A.append(x)
        for i in range(self.number_layer):
            input_layer = self.A[-1]   #ở ma trận A, A[-1] tức là lấy hàng cuối cùng
            z = np.dot(input_layer, self.W[i]) + self.b[i].T
            self.A.append(act_function(z, self.activation_function[i]))

    def back_propagation(self, x, t):
        """
        this function to calculate derivative of weight and bias of each layer
        :param x: input matrix x (nxm)
        :param t: output matrix t  (nx1)
        :return: NULL
        """
        # reset vector
        self.dJ_b = []  #Đạo hàm theo Bias
        self.dJ_W = []  #Đạo hàm theo Weight
        self.dJ_A = []  #Đạo hàm theo A
        # first: calculate dJ_dA(L)=2*(A(L)-t)   L = number_layer ; A(0) = x (nxm)
        #t = np.array(t).reshape(-1, 1)
        t = np.array(t)
        AL = self.A[-1]
        dJ_ZL = AL - t
        Al_subtract_1 = self.A[self.number_layer-1]
        #self.dJ_A.append(dJ_AL)
        # append in the list in the first time
        self.dJ_W.append(np.dot(Al_subtract_1.T, dJ_ZL))
        self.dJ_b.append(np.sum(dJ_ZL,0).reshape(-1, 1))
        self.dJ_A.append(np.dot(dJ_ZL,self.W[-1].T))

        # Calculate dJ_W(l),dJ_b(l),dJ_A(L-1)
        for i in reversed(range(0,(self.number_layer-1))):
            # define for step 0
            Al_subtract_1 = self.A[i]  # dim of A= number_layer + 1, but len(i) = number_layer  => A(l-1) = A[i]
            Al = self.A[i + 1]
            dJ_Al = self.dJ_A[-1]
            dAl_dZ = derivative_function(Al, self.activation_function[i])
            Wl = self.W[i]

            # append in the list
            self.dJ_W.append(np.dot(Al_subtract_1.T, dJ_Al * dAl_dZ))
            self.dJ_b.append(np.sum(dJ_Al * dAl_dZ, 0).reshape(-1, 1))
            self.dJ_A.append(np.dot((dJ_Al * dAl_dZ), Wl.T))

        # reversed dJ_W and dJ_b
        self.dJ_W = self.dJ_W[::-1]
        self.dJ_b = self.dJ_b[::-1]

    def update_weight(self):
        """
        this function to optimize weight and bias
        :return: NULL
        """

        for i in range(self.number_layer):
            #self.W[i] = self.W[i] - self.LR * self.dJ_W[i]
            #self.b[i] = self.b[i] - self.LR * self.dJ_b[i]
            self.W[i] = self.W[i] - self.LR*self.dJ_W[i] - self.LR*self.gama_momentum*self.momentum_chuanhan[i]
            self.b[i] = self.b[i] - self.LR*self.dJ_b[i] - self.LR*self.gama_momentum*self.momentumbias_chuanhan[i]
        self.momentum_chuanhan = self.dJ_W
        self.momentumbias_chuanhan = self.dJ_b

    def gradient_descent(self,x,t,batch_size):
        """
        this function to update weight and bias in one epoch
        :param x: input matrix x
        :param t: output matrix t
        :param batch_size: batch size, the data in final batch maybe < batch size
        :return: NULL
        """
        data_size = len(t)
        iterations = 0
        flag_batch_size = True
        if (data_size % batch_size) == 0:
            iterations = data_size/batch_size
        else:
            iterations = data_size//batch_size + 1
            flag_batch_size = False

        if flag_batch_size == True:
            for i in range(int(iterations)):
                in_data = x[i * batch_size:(i + 1) * batch_size]
                out_data = t[i * batch_size:(i + 1) * batch_size]

                self.feed_forward(in_data)
                self.back_propagation(in_data, out_data)
                self.update_weight()
        else:
            for i in range(int(iterations-1)):
                in_data = x[i * batch_size:(i + 1) * batch_size]
                out_data = t[i * batch_size:(i + 1) * batch_size]

                self.feed_forward(in_data)
                self.back_propagation(in_data, out_data)
                self.update_weight()
            in_data = x[int(iterations-1) * batch_size:]
            out_data = t[int(iterations-1) * batch_size:]
            self.feed_forward(in_data)
            self.back_propagation(in_data, out_data)
            self.update_weight()

    def y_predict(self,x):
        """
        this function to calculate y_predict
        :param x: input data matrix nxm
        :return: y_predict nx1
        """
        out_layer = np.array(x)
        for i in range(self.number_layer):
            z = np.dot(out_layer, self.W[i]) + self.b[i].T
            out_layer = act_function(z, self.activation_function[i])
        return out_layer.reshape(len(x),2)

    def E_rms_cross_entropy(self, x, t):
        """
        this function to calculate E_RMS
        :param x: input vector
        :param t: output vector
        :return: E_rms
        """
        y_predict = self.y_predict(x)
        E_rms = 0
        N = len(y_predict)
        for i in range(N):
            index = np.argmax(t[i])
            E_rms -= np.log(y_predict[i][index])
        return E_rms

    def E_Rate(self,x,t):
        '''
        This function is used for calculating Error of wrong results
        :param x: features input of data
        :param t: target/true label
        :return: E_rate
        '''
        y_predict = self.y_predict(x)
        y_predict_round = y_predict.round()
        target = t
        multiple = y_predict_round * target
        counter = np.sum(multiple)
        trueresults = len(target)-counter
        E_Rate = trueresults/len(target)
        return E_Rate*100

    def get_weight(self):
        return self.W
    def get_bias(self):
        return self.b



