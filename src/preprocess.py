import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

def normalize(x):
    return x.astype("float32") / 127.5 - 1.0

def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)["preprocess"]
    d = np.load("data/raw/fashion_mnist_raw.npz")
    x_train, y_train = normalize(d["x_train"]), d["y_train"]
    x_test, y_test = normalize(d["x_test"]), d["y_test"]
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=p["test_size"],
        random_state=p["seed"], stratify=y_train)
    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed("data/processed/fashion_mnist_processed.npz",
                        x_train=x_tr, y_train=y_tr, x_val=x_val, y_val=y_val,
                        x_test=x_test, y_test=y_test)
    print("Processed:", x_tr.shape, x_val.shape, x_test.shape)
    print("Class counts:", np.bincount(y_tr))

main()