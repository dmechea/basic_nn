import numpy as np

# Not using bias yet

# z1 = a(wz0 + b)
# w = weights
# z0 = previous activation (if none then input)
# b = bias
# z1 = output (or next layer)

# let pre_activation = w*z0 + b

# Using sigmoid activation
def sigmoid_activation(pre_activation):
    negative_matrix = -1 * pre_activation
    e_to_the_power_of_neg_x = np.exp(negative_matrix)
    denominator = 1 + e_to_the_power_of_neg_x
    return 1 / denominator

# Derivative of sigmoid
def sigmoid_derivative(sigmoid_output):
    return sigmoid_output*(1 - sigmoid_output)

# w * z0
def weights_multiplied_by_input(input_data, weight_layer): 
    return np.dot(input_data, weight_layer)

# z1 = a(w * z0)
def feed_forward(input_data, weights):
    pre_activation = weights_multiplied_by_input(input_data, weights)
    output = sigmoid_activation(pre_activation)
    return output

# Calculate a loss function
def squared_delta_loss(y_label, prediction):
    return (y_label - prediction)**2

# Derivative = 2 * (prediction - y_label) // Done by hand using 9th grade math
def squared_delta_loss_derivative(y_label, prediction):
    return 2 * (prediction - y_label)

# Dummy input data
x = np.array( [[1, 1, 0], [1, 0, 0],[1, 1, 1], [0, 0, 0], [1,0,0]] )

# Dummy label data
y = [0, 1, 1, 0, 0]

# Weight Layer
weight_layer_1 = 2* np.random.random( (3,1) ) - 1

# Learning Rate
learning_rate = 0.0001

# Training Loop
for iteration in range(100000):
    
    # Prediction
    pred = feed_forward(x, weight_layer_1)

    # Loss function
    loss = squared_delta_loss(y, pred)

    # gradient of prediction output on the Loss function
    dL_dp = squared_delta_loss_derivative(y, pred).T

    # Local gradient of activation function 
    dp_da = sigmoid_derivative(pred) # pred is the output of the sigmoid activation function.

    # Chain rule to get pre activation function gradient
    dL_da = np.dot(dL_dp, dp_da).T # Chain Rule to get derivative

    # Local Gradients of the weights with respect to pre_activation output
    da_dw = x # It's just a multiplication so local gradient is equal to input

    # Gradients that the weights have with respect to output Loss (a.k.a what we want to calculate)
    dL_dw = np.dot(dL_da, da_dw).T

    # Logging
    if iteration % 10000 == 0:
        print (loss)
        print (dL_dw)

    # Standard gradient descent => Update weights
    weight_layer_1 += -dL_dw * learning_rate

