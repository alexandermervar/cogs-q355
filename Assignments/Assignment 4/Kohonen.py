import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from math import dist

fig, ax = plt.subplots(figsize=(1,1))

# Define Learning Rate
learningRate = 0.01

# Define weight matrix size
# TODO: Change to fit assignment
xSize = 25
ySize = 25
weightVectorSize = 3

# Define Frequency of Input Vectors
# TODO: Change to fit assignment
redFrequency = 1/3
greenFrequency = 2/3
blueFrequency  = 0

# Define the number of iterations
# TODO: Change to fit assignment
iterations = 100


# Define an random X*Y*WV array
weightMatrix = np.random.random((xSize, ySize, weightVectorSize))

for i in range(iterations):
    # Define a random input vector
    inputVectorSelector = random.random()

    # Create input vector
    if inputVectorSelector < redFrequency:
        inputVector = [1, 0, 0]
    elif inputVectorSelector < greenFrequency:
        inputVector = [0, 1, 0]
    else:
        inputVector = [0, 0, 1]

    # Traverse the matrix
    for j in range(xSize):
        #
        for k in range(ySize):
            # Select the SOFM Unit
            selectedUnit = weightMatrix[j,k]

            # Calculate the external input (The dot product of the input vector and the weight vector)
            externalInput = np.dot(inputVector, selectedUnit)

            # Traverse to every other unit
            for l in range(xSize):
                for m in range(ySize):
                    # Select a new SOFM Unit
                    selectedUnit = weightMatrix[l,m]

                    # Calculate eta
                    # TODO: Complete this
                    eta = 

                    # TODO: Add wrap-around mechanic
                    # Calculate the lateral input
                    if (dist(j,k,l,m) < 3):
                        lateralInput = eta * 8
                    elif (8 > dist(j,k,l,m) > 3):
                        lateralInput = eta * -1
                    else:
                        lateralInput = 0
# Produce an Image
plt.imshow(weightMatrix)

plt.show()

# Commented out because I don't have "count" defined
# ax.set_title('%f'%(count))

fig.canvas.draw()

fig.canvas.flush_events()