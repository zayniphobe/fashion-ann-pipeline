import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras

def main():
    d = np.load("data/processed/fashion_mnist_processed.npz")
    model = keras.models.load_model("models/model.h5")
    loss, acc = model.evaluate(d["x_test"], d["y_test"], verbose=0)
    pred = model.predict(d["x_test"], verbose=0).argmax(axis=1)
    cm = confusion_matrix(d["y_test"], pred)
    os.makedirs("reports", exist_ok=True)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.savefig("reports/confusion_matrix.png", dpi=150, bbox_inches="tight")
    with open("metrics.json", "w") as f:
        json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)
    print("test_loss", loss, "test_accuracy", acc)

main()