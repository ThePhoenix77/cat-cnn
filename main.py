import numpy as np
from PIL import Image
from cnn_network import GradientDescent
from utils import bce
from cat_cnn import CatCNN

import os

def load_data(folder_cat, folder_noncat, img_size=(64, 64)):
    from PIL import Image
    X, y = [], []
    for filename in os.listdir(folder_cat):
        img = Image.open(os.path.join(folder_cat, filename)).convert("L")
        img = img.resize(img_size)
        X.append(np.array(img) / 255.0)
        y.append(1)
    for filename in os.listdir(folder_noncat):
        img = Image.open(os.path.join(folder_noncat, filename)).convert("L")
        img = img.resize(img_size)
        X.append(np.array(img) / 255.0)
        y.append(0)
    return np.array(X), np.array(y)

def train_cnn(cnn, X_train, y_train, epochs=5, lr=0.01):
    optimizer = GradientDescent(lr)
    for epoch in range(epochs):
        total_loss = 0
        for x, y in zip(X_train, y_train):
            y_pred = cnn.forward(x)
            loss = bce(y, y_pred)
            grads = cnn.backward(y, x)
            optimizer.update({"kernel": cnn.kernel, "dense_W": cnn.dense.W, "dense_b": cnn.dense.b}, grads)
            total_loss += loss
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(X_train)}")

def predict_image(cnn, image_path):
    img = Image.open(image_path).convert("L").resize((64, 64))
    x = np.array(img) / 255.0
    y_pred = cnn.forward(x)
    print("Cat" if y_pred >= 0.5 else "Not a cat")

def main():
    cnn = CatCNN()
    print("Loading data...")
    X_train, y_train = load_data("data/cats", "data/noncats")
    print("Training CNN...")
    train_cnn(cnn, X_train, y_train)

    while True:
        img_path = input("Enter image path: ")
        predict_image(cnn, img_path)

if __name__ == "__main__":
    main()
