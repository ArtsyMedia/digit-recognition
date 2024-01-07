from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np


def view(a):
    x = a.reshape(8, 8)
    plt.imshow(x, cmap='binary')
    plt.show()


digits = load_digits()

y = digits.target
x = digits.images.reshape((len(digits.images), -1))

mlp = MLPClassifier(hidden_layer_sizes=(15,), activation='logistic', alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1, verbose=True)

mlp.fit(x, y)

def Predict(data):
    prob_data = mlp.predict_proba(data.reshape((1, 64)))[0]

    confidences = {}
    keys = range(10)
    for i in keys:
        prob_iteration = prob_data[i] * 100
        confidences[i] = prob_iteration
    return confidences
