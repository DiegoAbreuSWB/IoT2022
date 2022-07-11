import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# AUxillary methods
def getDist(y):
    ax = sns.countplot(y)
    ax.set(title="Count of data classes")
    plt.show()


def getData(dist, x, y):
    dx = []
    dy = []
    counts = [0 for i in range(10)]
    for i in range(len(x)):
        if counts[y[i]] < dist[y[i]]:
            dx.append(x[i])
            dy.append(y[i])
            counts[y[i]] += 1

    return np.array(dx), np.array(dy)


# Load and compile Keras model
model = keras.Sequential([
    #keras.layers.Flatten(input_shape=(28, 28)),
     keras.layers.Flatten(input_shape=(74,1)), #74 features, de uma dimensão o array
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(2, activation='softmax')

])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

data = pd.read_csv (r'C:\Users\diego\PycharmProjects\MNIST-Federated\03_Non IID Demo\bases\12.luohe_binario.csv')#binario



#y= data.Tipo #multiclass
y= data.Comportamento #binario
print(y)
x = data.drop('Comportamento', axis=1)
y= np.array(y)
x= np.array(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), verbose=0)
        hist = r.history
      #  print("Fit history : ", hist)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

        print("Eval accuracy : ", accuracy)

        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="localhost:" + str(sys.argv[1]),
    client=FlowerClient(),
   # grpc_max_message_length=1024 * 1024 * 1024
)