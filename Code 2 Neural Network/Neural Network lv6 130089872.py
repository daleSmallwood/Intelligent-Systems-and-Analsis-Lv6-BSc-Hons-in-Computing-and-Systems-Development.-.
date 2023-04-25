import sys
import numpy as np
import matplotlib
import random


#neural network input layer
X = random.randrange(2, 20)
y = random.randrange(1,40)
#hidden layer
class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
	def foward(self, inputs):
		self.output = np.dot(inputs,self.weights) + self.biases



#activation function ReLU activationb

class Activation_ReLU:
	def foward(self, inputs):
		self.output = np.maximum(0, inputs)


#activation function Softmax
class Activation_Sofatmax:
	def foward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities


#Loss function
class Loss:
	def calculate(self, output, y):
		sample_losses = self.foward(output, y)
		data_loss = np.mean(sample_losses)
		return data_loss

class Loss_CategoricalCrossentropy(Loss):
	def foward(self, y_pred, y_true):
		samples = len(y_pred)
		y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

		if len(y_true.shape) == 1:
			correct_confidences = y_pred_clipped[range(samples), y_true]
		
		elif len(y_true.shape) == 2:
			correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

		negative_log_likelihoods = -np.log(correct_confidences)
		return negative_log_likelihoods

#layer 1
dense1 = Layer_Dense (2,3)
activation1 = Activation_ReLU()

#layer 2 
dense2 = Layer_Dense(3,3)
activation2 = Activation_Sofatmax()

dense1.foward(X)
activation1.foward(dense1.output)


dense2.foward(activation2.output)
activation2.foward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)








#print(0.10*np.random.randn(4, 3))




'''
inputs = [1, 2, 3, 2.5]#single sample set
'''

'''
inputs = [[1, 2, 3, 2.5],
		 [2.0, 5.0, -1.0, 2.0],
		 [-1.5, 2.7, 3.3, -0.8]]         


#batch sample set helping with the fitment line 



weights = [[0.2, 0.8, -0.5, 1.0],
		   [0.5, -0.91, 0.26, -0.5],
		   [-0.26, 0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]
#layer1

weights2 = [[0.1, -0.14, 0.5],
		   [-0.5, 0.12, -0.33],
		   [-0.44, 0.73, -0.13]]

biases2 = [-1,2, -0.5]
#layer2

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)

print(output)
'''

'''
weights = [0.2, 0.8, -0.5, 1.0]
		   
bias = [2]

output = np.dot(weights, inputs)+ bias
print(output)
'''

'''
some_value = 0.5
weight = 0.7
bias = 0.7

print(some_value*weight)
print(some_value+bias)

'''

'''
layer_outputs = [] #Ouput of current layer
for neuron_weights, neruon_bias in zip(weights, biases):
	neuron_output = 0 # Output of given neuron
	for n_input, weight in zip(inputs, neuron_weights):
		neuron_output += n_input*weight
	neuron_output += neruon_bias
	layer_outputs.append(neuron_output)

print(layer_outputs)
'''

'''
output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
		  inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,		
		  inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
print(output)
'''


