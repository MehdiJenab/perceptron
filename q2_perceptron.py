

# developed by Mehd Jenab
# mehdi.jenab@ryerson.ca
# This can handle two type of input for data points, 
# 1. within a box by useBoxes=True
# 2. produced by make_blobs by useCenters=True
# First option produce points in a box and second one produce 
# data points scattered around given centers.


import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

class perceptronClass():
	def __init__(self, *arg, **kwargs): 
			self.n_features		= kwargs.get('n_features', 0)
			self.epochs 		= kwargs.get('epochs', 10)
			self.learning_rate 	= kwargs.get('learning_rate', 0.01)
			self.weights		= np.zeros(self.n_features)
			self.bias			= 0.0
			self.error 			= -1.0
	
	def predict(self, vector):
		value = np.dot(vector, self.weights) + self.bias
		return 1 if value > 0 else 0


	def train(self, training_inputs, labels):
		nTrain = len(training_inputs)
		for _ in range(self.epochs):
			#print
			for i in range(nTrain):
				prediction	= self.predict(training_inputs[i])
				error 		= labels[i] - prediction
				#print i, self.weights, self.bias, training_inputs[i],labels[i],prediction, error
				for j in range(self.n_features):
					self.weights[j]	+= self.learning_rate * error *training_inputs[i][j]
				self.bias		+= self.learning_rate * error
				
	
	def test(self, test_vector, labels):
		nTest = len(test_vector)
		errors = []
		for i in range(nTest):
			prediction	= self.predict(test_vector[i])
			errors.append(labels[i] - prediction)
			#print labels[i] , prediction
		self.error = np.mean(errors)

	def print_results(self):
		print "error=", self.error
		print  "weights=",list(self.weights), ", bias=", self.bias

class samplesInBox():
	def __init__(self, *arg, **kwargs):
		box				= kwargs.get('box', [[0,1],[0,1],[0,1]]) # should be given in [[],[],[]]
		n_sample		= kwargs.get('n_sample', 10)
		label			= kwargs.get('label', -1)
		
		self.labelSet 	= [label] * n_sample
		self.dataSet	= []
		

		for _ in range(n_sample):
			data = []
			for axis in box:
				data.append(np.random.uniform(axis[0], axis[1],1)[0])
			self.dataSet.append(data)

class divide_train_test():
	def __init__(self,dataSet,labelSet, *arg, **kwargs):
		n_sample	= kwargs.get('n_sample', 0) 
		
		dataLabelSet = zip(dataSet,labelSet)
		np.random.shuffle(dataLabelSet)
		self.dataSet,self.labels =map(list, zip(*dataLabelSet))
		
		trainInterval = [0,100]#[0,int(0.5*n_sample)]
		self.dataTrainSet,self.labelTrainSet = self.getSet(trainInterval)
		
		testInterval = [1000,2000]#[int(0.5*n_sample)+1,n_sample]
		self.dataTestSet,self.labelTestSet   = self.getSet(testInterval)
		
	def getSet(self,interval):		
		dataSet = self.dataSet[interval[0]:interval[1]]
		labelSet = self.labels[interval[0]:interval[1]]
		
		return np.array(dataSet), np.array(labelSet)

	def get_train_test(self):
		return self.dataTrainSet, self.labelTrainSet, self.dataTestSet, self.labelTestSet

class make_blobs_in_boxes():
	def __init__(self, *arg, **kwargs):
		boxes		= kwargs.get('boxes', 0) 
		n_samples	= kwargs.get('n_samples', 10)
		n_features	= kwargs.get('n_features', 0)
		labels = [0,1]
		
		self.dataSet = []
		self.labelSet = []
		for i in range(2):
			dataBox   = samplesInBox(box=boxes[i],n_sample=n_samples[i],label=labels[i])
			self.dataSet  += dataBox.dataSet
			self.labelSet += dataBox.labelSet


	def get_data_label(self):
		return self.dataSet,self.labelSet


# drive part of the code

#==============================================================================
# 1. initialization of parameters

useBoxes = False
useCenters = True

n_sample = 4000
n_features = 3
np.random.seed(1)

n_sample = int(n_sample/2) # in order to have equal number of each class
n_samples = [n_sample,n_sample]
n_sample = n_samples[0]+n_samples[1]

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")


if useBoxes:	
	box0 	= [[14, 20], [0 , 6 ], [ 0,  6]]
	box1 	= [[0, 6  ], [14, 20], [14, 20]]
	
	boxes = [box0,box1]
	blobs = make_blobs_in_boxes(boxes=boxes, n_samples=n_samples, n_features=n_features)
	dataSet, labelSet = blobs.get_data_label()

if useCenters:
	centers = [(17, 3, 3), (3, 17, 17)]
	dataSet, labelSet = make_blobs(n_samples=n_sample, centers=centers, cluster_std=4.0, n_features=n_features)
	ax.scatter3D(dataSet[:, 0], dataSet[:, 1], dataSet[:, 2], marker="s", c = labelSet)
#==============================================================================



#==============================================================================
# 2. produce train and test sets
obj = divide_train_test(dataSet, labelSet,n_sample=n_sample)
dataTrainSet,labelTrainSet, dataTestSet,labelTestSet = obj.get_train_test() 
#==============================================================================


#==============================================================================
# 3. perceptron
# 3.1. create the perceptron
perceptron = perceptronClass(n_features=3, epochs =5, learning_rate=1.0)

# 3.2. train the perceptron
perceptron.train(dataTrainSet,labelTrainSet)

# 3.3. test the perceptron
perceptron.test(dataTestSet,labelTestSet)

#==============================================================================




#==============================================================================
# 4. print the results and visualiztion 
perceptron.print_results()


weights = perceptron.weights
bias = perceptron.bias
x1, x2 = np.meshgrid(range(30), range(30))
x3 = -(weights[0]*x1 + weights[1]*x2 + bias)/weights[2]

# plot the plane
ax.plot_surface(x1, x2, x3, alpha=0.5)

ax.set_xlabel('X ')
ax.set_ylabel('Y ')
ax.set_zlabel('Z ')
plt.show()
#==============================================================================

