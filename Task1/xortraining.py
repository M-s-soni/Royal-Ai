import numpy as np
from Xor import Perceptron

training_inputs=[]
training_inputs.append(np.array([1,1]))
training_inputs.append(np.array([1,0]))
training_inputs.append(np.array([0,1]))
training_inputs.append(np.array([0,0]))

labels = np.array([0,1,1,0])

xor = Perceptron(2)
xor.train(training_inputs,labels)

inputs=np.array([1,1])
print(xor.predict(inputs))