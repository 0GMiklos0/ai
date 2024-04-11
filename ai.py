import numpy as np

input_vector = [1.66, 1.56]
weights = [1.45, -0.66]
bias = [0.0]

first_index = input_vector[0] * weights[0]
second_index = input_vector[1] * weights[1]

dot_product = first_index + second_index
print("skalaris szorzat:",dot_product)

dot_product2 = np.dot(input_vector, weights)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print(sigmoid(0))

def make_prediction(input, weight, bias):
    layer_1_result = np.dot(input, weight) + bias
    layer_2_result = sigmoid(layer_1_result)

    return layer_2_result

prediction = make_prediction(input_vector,weights, bias)

target = 0

se = np.square(prediction-target)

print("a kovetkeztetes:", prediction, ", a hiba:", se)

derivate = 2 * (prediction - target) 

print("derivalt:", derivate)

weights = weights - derivate
prediction = make_prediction(input_vector, weights, bias)
error = np.square(prediction-target)

print("kovetkeztetes:", prediction, "A hiba:", error)