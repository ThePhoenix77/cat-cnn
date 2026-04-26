import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cat_cnn import CatCNN
from main import load_model


LAYER_SAMPLE_SIZES = {
    "input": 20,
    "conv1": 24,
    "pool1": 20,
    "conv2": 18,
    "pool2": 14,
    "flatten": 24,
    "output": 1,
}


def load_input_image(image_path):
    img = Image.open(image_path).convert("L").resize((64, 64))
    arr = np.array(img) / 255.0
    return arr


def sample_array(values, count):
    flat = np.asarray(values, dtype=float).reshape(-1)
    if len(flat) <= count:
        indices = np.arange(len(flat), dtype=int)
        sampled = flat
    else:
        indices = np.linspace(0, len(flat) - 1, count, dtype=int)
        sampled = flat[indices]

    return sampled.astype(float), indices.astype(int)


def layer_payload(name, values, count):
    sampled_values, sampled_indices = sample_array(values, count)
    arr = np.asarray(values, dtype=float)
    return {
        "name": name,
        "shape": list(arr.shape),
        "sampled_values": sampled_values.tolist(),
        "sampled_indices": sampled_indices.tolist(),
        "stats": {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Export CatCNN activations for animated network visualization")
    parser.add_argument("image_path", type=str, help="Path to an input image")
    parser.add_argument("--model-path", type=str, default="final_cnn_model.npz", help="Saved model weights file")
    parser.add_argument(
        "--output-json",
        type=str,
        default="visualizations/network_animation_data.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    cnn = CatCNN()
    load_model(cnn, args.model_path)

    x = load_input_image(args.image_path)
    prediction = float(np.squeeze(cnn.forward(x)))
    predicted_label = "cat" if prediction >= 0.5 else "not-cat"

    layers = {
        "input": x,
        "conv1": cnn.conv1,
        "pool1": cnn.pool1,
        "conv2": cnn.conv2,
        "pool2": cnn.pool2,
        "flatten": np.squeeze(cnn.flatten),
        "output": np.array([prediction]),
    }

    payload_layers = []
    flatten_indices = None

    for name in ["input", "conv1", "pool1", "conv2", "pool2", "flatten", "output"]:
        layer = layer_payload(name, layers[name], LAYER_SAMPLE_SIZES[name])
        payload_layers.append(layer)
        if name == "flatten":
            flatten_indices = np.array(layer["sampled_indices"], dtype=int)

    dense_weights = cnn.dense.W.reshape(-1)
    sampled_dense_weights = dense_weights[flatten_indices].astype(float)

    output = {
        "meta": {
            "image_path": args.image_path,
            "model_path": args.model_path,
            "prediction": prediction,
            "predicted_label": predicted_label,
            "threshold": 0.5,
        },
        "layers": payload_layers,
        "dense": {
            "sampled_weights": sampled_dense_weights.tolist(),
            "bias": float(np.squeeze(cnn.dense.b)),
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Saved animation data to {output_path}")
    print(f"Prediction: {predicted_label} ({prediction:.4f})")


if __name__ == "__main__":
    main()
