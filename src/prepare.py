import os
import ssl
import certifi
import numpy as np
from tensorflow import keras

# Instruct SSL context to use certifi's bundle
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    os.makedirs("data/raw", exist_ok=True)
    np.savez_compressed("data/raw/fashion_mnist_raw.npz",
                        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print("Saved raw data:", x_train.shape, x_test.shape)

main()