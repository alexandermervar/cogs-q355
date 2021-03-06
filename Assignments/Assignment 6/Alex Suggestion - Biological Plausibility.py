import numpy as np

learningRate = 0.001

def nonlin(x, deriv=False):
    if (deriv==True):
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))

# input dataset as a matrix for the XOR problem
x = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])

# output dataset as a matrix for the XOR problem
y = np.array([[0], [1], [1], [0]])

np.random.seed(1)

# synapse matrices
forwardSyn0 = 2*np.random.random((3,10)) - 1
forwardSyn1 = 2*np.random.random((10,1)) - 1
# New fixed backward weights with SMALL and RANDOM values
backwardSyn0 = 2*np.random.random((1,10)) - 1

# training step
for i in range(100000):

    fl0 = x
    fl1 = nonlin(np.dot(fl0, forwardSyn0))
    fl2 = nonlin(np.dot(fl1, forwardSyn1))

    l2_error = y - fl2

    # Feedback Alignment Network Nodes
    bl0 = fl2
    bl1 = nonlin(np.dot(bl0, backwardSyn0))

    if(i % 10000) == 0:
        print("Feedback Alignment Error: " + str(np.mean(np.abs(l2_error))))

    # ========================
    # Old Backpropagation Code
    # oldl2_delta = l2_error*nonlin(fl2, deriv=True)

    # oldl1_error = l2_delta.dot(forwardSyn1.T)

    # oldl1_delta = l1_error * nonlin(fl1,deriv=True)
    # ======================
    # update synapse weights

    error = y - bl0

    l2_delta = error * nonlin(bl0, deriv=True)
    l1_error = l2_delta.dot(backwardSyn0)
    l1_delta =  l1_error * nonlin(bl1,deriv=True)

    forwardSyn1 += fl1.T.dot(l2_delta) * learningRate
    forwardSyn0 += fl0.T.dot(l1_delta) * learningRate

print("FEEDBACK ALIGNMENT - Output after training:")
print(fl2)