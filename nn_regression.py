import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import warnings
import argparse
import os
warnings.filterwarnings('ignore')


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def sigmoid_gradient(a):
    return (1 - sigmoid(a)) * (sigmoid(a))


def forward(X):
    Hidden = X.dot(W1)
    # Hidden_sig = sigmoid(Hidden)
    Hidden_sig = np.tanh(Hidden)
    Out = Hidden_sig.dot(W2)
    np.around(Out, decimals=2)
    return Hidden_sig, Out


def gradient_W2(Hidden_sig, Y, Out):
    # return (Y - Out).dot(Hidden_sig)
    return (Out - Y).dot(Hidden_sig)


def gradient_W1(X, Hidden_sig, Y, Out, W2):
    # dZ = np.outer(Y - Out, W2) * sigmoid_gradient(Hidden_sig)
    dZ = np.outer(Out - Y, W2) * (1 - Hidden_sig * Hidden_sig)  # tanh derivative
    dZ = X.transpose().dot(dZ)
    return dZ


def learn(X, Hidden_sig, Y, Out, W1, W2, learning_rate=0.005):
    dW2 = gradient_W2(Hidden_sig, Y, Out)
    dW1 = gradient_W1(X, Hidden_sig, Y, Out, W2)

    W2 -= learning_rate * dW2
    W1 -= learning_rate * dW1

    return W1, W2


def get_squared_error(Y, Out):
    c = np.square(Y - Out)
    return c.mean()  # .sum() / N


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-layer",
                        type=int,
                        default=2,
                        help="value for input layer length")
    parser.add_argument("-H", "--hidden-layer",
                        type=int,
                        default=50,
                        help="value for hidden layer length")
    parser.add_argument("-f", "--file", required=True, help="dataset file")
    args = parser.parse_args()
    print(f"Opening: {args.file}")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), args.file))

    X = np.array([df.x, df.y]).transpose()
    Y = np.array(df.z)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Y, c='red')
    plt.show()


    D = args.input_layer
    M = args.hidden_layer

    W1 = np.random.rand(D, M)
    W2 = np.random.rand(M)
    
    costs = []
    Hidden_sig, Out = forward(X)
    cost = get_squared_error(Y, Out)
    print(cost)
    i = 0
    while cost > 0.1:
        Hidden_sig, Out = forward(X)
        W1, W2 = learn(X, Hidden_sig, Y, Out, W1, W2)
        cost = get_squared_error(Y, Out)
        costs.append(cost)
        if i % 25 == 0:
            print(cost)
        i += 1
    print(f"Took {i} iterations")
    plt.plot(costs)
    plt.show()

    line = np.linspace(0, 2, 50)
    xx, yy = np.meshgrid(line, line)

    # Create a new dataset for Keras model
    K = np.vstack((xx.flatten(), yy.flatten())).T

    _, Out = forward(K)
    model = MLPRegressor(hidden_layer_sizes=(50,),
                         activation='tanh',
                         solver='adam',
                         alpha=0.01,
                         max_iter=9500)

    df = pd.read_csv(args.file)
    X = np.array([df.x, df.y]).transpose()
    Y = np.array(df.z)
    model.fit(X, Y)
    # Prediction of Keras model
    Out2 = model.predict(K)

    # surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(K[:, 0], K[:, 1], Out, linewidth=0.2, antialiased=True, alpha=0.5)
    ax.plot_trisurf(K[:, 0], K[:, 1], Out2, linewidth=0.2, antialiased=True, color='red', alpha=0.7)
    plt.show()
