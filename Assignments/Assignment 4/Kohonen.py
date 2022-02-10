import random
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import colors

fig, ax = plt.subplots(figsize=(10,10))

# Define Learning Rate
learningRate = 0.01

# Define weight matrix size
# TODO: Change to fit assignment
xSize = 50
ySize = 50
weightVectorSize = 3
mapSize = xSize

# Define Frequency of Input Vectors
# TODO: Change to fit assignment
redFrequency = 1/3
greenFrequency = 2/3
blueFrequency  = 0

# Define the number of iterations
# TODO: Change to fit assignment
iterations = 50


# Define an random X*Y*WV array
weightMatrix = np.random.random((xSize, ySize, weightVectorSize))

for i in range(iterations):
    # Define a random input vector
    inputVectorSelector = random.random()

    # Create input vector
    if inputVectorSelector < redFrequency:
        inputVector = np.array([1, 0, 0])
    elif inputVectorSelector < redFrequency + greenFrequency:
        inputVector = np.array([0, 1, 0])
    else:
        inputVector = np.array([0, 0, 1])

    # Traverse the matrix
    for j in range(xSize):
        for k in range(ySize):
            # Select the SOFM Unit
            selectedUnit = weightMatrix[j,k]

            # Calculate the external input (The dot product of the input vector and the weight vector)
            externalInput = np.dot(inputVector, selectedUnit)

            tmp = externalInput

            # Traverse to every other unit
            for l in range(xSize):
                for m in range(ySize):
                    if (i == l & j == m):
                        continue
                    else:
                        # TODO: Fix wrap around mechanic
                        # Select a new SOFM Unit
                        # xval and yval are the coordinates of the current units
                        for p in range(-8,8):
                            for q in range(-8,8):
                                # TODO: What is this?
                                # edist = np.sqrt(p**2 + q**2)
                                modx = np.mod(j + p, mapSize)
                                mody = np.mod(k + q, mapSize)
                                otherUnit = weightMatrix[modx, mody]

                                # Calculate lateral input
                                if (np.dot(inputVector, otherUnit) < 3):
                                    lateralInput = np.dot(inputVector, otherUnit) * 8
                                elif (3 > np.dot(inputVector, otherUnit) < 8):
                                    lateralInput = np.dot(inputVector, otherUnit) * -1
                                else:
                                    lateralInput = 0

                                # Add both the external inputs and neighbor inputs into a new variable
                                tmp += lateralInput
                        
            # Calcualte eta for tmp
            if (tmp <= 0):
                eta = 0
            elif (tmp >= 5):
                eta = 5
            else:
                eta = tmp

            # Calculate the new weight vector for the selected unit
            weightMatrix[j,k] = (selectedUnit + (learningRate * eta * inputVector))/np.linalg.norm(inputVector - selectedUnit)

# Produce an Image
plt.imshow(weightMatrix)

plt.show()

ax.set_title("Mervar Kohonen Map")

fig.canvas.draw()

fig.canvas.flush_events()