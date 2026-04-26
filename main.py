import numpy as np
import argparse
import json
from PIL import Image
from cnn_network import GradientDescent
from utils import bce
from cat_cnn import CatCNN
from data_loader import load_data

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

def evaluate_metrics(cnn, X_data, y_data):
    y_probs = []
    losses = []

    for x, y in zip(X_data, y_data):
        y_pred = float(np.squeeze(cnn.forward(x)))
        y_probs.append(y_pred)
        losses.append(float(bce(y, y_pred)))

    y_probs = np.array(y_probs)
    y_true = np.array(y_data).astype(int)
    y_pred_labels = (y_probs >= 0.5).astype(int)

    tp = int(np.sum((y_pred_labels == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred_labels == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred_labels == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred_labels == 0) & (y_true == 1)))

    accuracy = float((tp + tn) / len(y_true)) if len(y_true) else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "samples": int(len(y_true)),
    }

def save_metrics(metrics, metrics_path):
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"Saved metrics to {metrics_path}")

def print_metrics(metrics):
    print("Training metrics:")
    print(f"  Loss:      {metrics['loss']:.6f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  TP/TN/FP/FN: {metrics['tp']}/{metrics['tn']}/{metrics['fp']}/{metrics['fn']}")

def save_model(cnn, save_path):
    np.savez(
        save_path,
        kernel1=cnn.kernel1,
        kernel2=cnn.kernel2,
        dense_W=cnn.dense.W,
        dense_b=cnn.dense.b,
    )
    print(f"Saved model to {save_path}")

def load_model(cnn, model_path):
    params = np.load(model_path)
    cnn.kernel1 = params["kernel1"]
    cnn.kernel2 = params["kernel2"]
    cnn.dense.W = params["dense_W"]
    cnn.dense.b = params["dense_b"]
    print(f"Loaded model from {model_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train and run CatCNN")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data-cats", type=str, default="data/cats")
    parser.add_argument("--data-noncats", type=str, default="data/noncats")
    parser.add_argument("--save-path", type=str, default="final_cnn_model.npz")
    parser.add_argument("--metrics-path", type=str, default="training_metrics.json")
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--train-only", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    cnn = CatCNN()

    if args.load_model:
        load_model(cnn, args.load_model)

    print("Loading data...")
    X_train, y_train = load_data(args.data_cats, args.data_noncats)
    print("Training CNN...")
    train_cnn(cnn, X_train, y_train, epochs=args.epochs, lr=args.lr)

    metrics = evaluate_metrics(cnn, X_train, y_train)
    print_metrics(metrics)
    save_metrics(metrics, args.metrics_path)

    save_model(cnn, args.save_path)

    if args.train_only:
        return

    while True:
        try:
            img_path = input("Enter image path: ")
            predict_image(cnn, img_path)
        except EOFError:
            print("Exiting prediction loop.")
            break

if __name__ == "__main__":
    main()
