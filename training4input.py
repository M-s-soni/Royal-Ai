import numpy as np
from Perceptroninput4 import Perceptron

training_inputs=[]
training_inputs.append(np.array([1,1,1,1]))
training_inputs.append(np.array([1,1,1,0]))
training_inputs.append(np.array([0,1,1,1]))
training_inputs.append(np.array([1,1,0,0]))
training_inputs.append(np.array([0,0,1,1]))
training_inputs.append(np.array([1,0,0,1]))

labels=np.array([1,1,1,0,0,0])

data = Perceptron(4)
data.train(training_inputs,labels)

inputs = np.array([1,0,0,1])
print(data.predict(inputs))