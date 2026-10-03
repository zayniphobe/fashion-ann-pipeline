import csv
import os
import numpy as np
import yaml
from tensorflow import keras

def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)["train"]
    d = np.load("data/processed/fashion_mnist_processed.npz")
    model = keras.Sequential([
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(p["dense_units"], activation="relu"),
        keras.layers.Dropout(p["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(d["x_train"], d["y_train"],
                     validation_data=(d["x_val"], d["y_val"]),
                     epochs=p["epochs"], batch_size=p["batch_size"]).history
    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")
    with open("models/history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + list(hist))
        for i in range(len(hist["loss"])):
            w.writerow([i + 1] + [hist[k][i] for k in hist])

main()