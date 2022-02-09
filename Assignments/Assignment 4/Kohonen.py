import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from math import dist

fig, ax = plt.subplots(figsize=(10,10))

# Define Learning Rate
learningRate = 0.01

# Define weight matrix size
# TODO: Change to fit assignment
xSize = 10
ySize = 10
weightVectorSize = 3

# Define Frequency of Input Vectors
# TODO: Change to fit assignment
redFrequency = 1/3
greenFrequency = 2/3
blueFrequency  = 0

# Define the number of iterations
# TODO: Change to fit assignment
iterations = 1


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
        for k in range(ySize):
            # Select the SOFM Unit
            selectedUnit = weightMatrix[j,k]

            # Calculate the external input (The dot product of the input vector and the weight vector)
            externalInput = np.dot(inputVector, selectedUnit)

            # Calculate the lateral input for the selected unit
            lateralInputSelectedUnit = externalInput * 8

            # Calculate the eta of the selcted unit
            if (lateralInputSelectedUnit + externalInput <= 0):
                eta = 0
            elif (lateralInputSelectedUnit + externalInput >= 5):
                eta = 5
            else:
                eta = lateralInputSelectedUnit + externalInput


            # Calculate the new weight vector for the selected unit
            for l in range(weightVectorSize):
                weightMatrix[j,k,l] = (weightMatrix[j,k,l] + learningRate * eta * (lateralInputSelectedUnit + externalInput))/dist(inputVector, selectedUnit)

            # Traverse to every other unit
            for l in range(xSize):
                for m in range(ySize):
                    if (i == l & j == m):
                        continue
                    else:
                        # Select a new SOFM Unit
                        selectedUnit2 = weightMatrix[l,m]

                        # TODO: Add wrap-around mechanic
                        # Calculate the lateral input
                        if (dist((j,k), (l,m)) < 3):
                            lateralInput = eta * 8
                        elif (8 > dist((j,k), (l,m)) > 3):
                            lateralInput = eta * -1
                        else:
                            lateralInput = 0

                        # Calculate the external input
                        externalInput2 = np.dot(weightMatrix[j,k], selectedUnit2)

                        # Calculate the change in weights
                        for n in range(weightVectorSize):
                            weightMatrix[l,m,n] = (weightMatrix[l,m,n] + learningRate * eta * (lateralInput + externalInput2))/dist(inputVector, selectedUnit2)

# Produce an Image
plt.imshow(weightMatrix)

plt.show()

ax.set_title("Mervar Kohonen Map")

fig.canvas.draw()

fig.canvas.flush_events()