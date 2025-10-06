import numpy as np
from PIL import Image
from cnn_network import GradientDescent
from utils import bce
from cat_cnn import CatCNN
from data_loader import load_data

import os

def train_cnn(cnn, X_train, y_train, epochs=10, lr=0.01):
    optimizer = GradientDescent(lr)
    for epoch in range(epochs):
        total_loss = 0
        for x, y in zip(X_train, y_train):
            y_pred = cnn.forward(x)
            loss = bce(y, y_pred)
            grads = cnn.backward(y, x)

            params = {
                "kernel1": cnn.kernel1,
                "kernel2": cnn.kernel2,
                "dense_W": cnn.dense.W,
                "dense_b": cnn.dense.b
            }
            optimizer.update(params, grads)
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
