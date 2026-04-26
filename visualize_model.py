import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cat_cnn import CatCNN
from main import load_model


def normalize_array(arr):
    arr = np.array(arr, dtype=np.float32)
    arr_min = float(arr.min())
    arr_max = float(arr.max())

    if np.isclose(arr_max, arr_min):
        return np.zeros_like(arr, dtype=np.uint8)

    scaled = (arr - arr_min) / (arr_max - arr_min)
    return (scaled * 255).astype(np.uint8)


def save_heatmap(array, path, resize_to=None):
    img = Image.fromarray(normalize_array(array), mode="L")
    if resize_to is not None:
        img = img.resize(resize_to, Image.Resampling.NEAREST)
    img.save(path)


def load_input_image(image_path):
    img = Image.open(image_path).convert("L").resize((64, 64))
    arr = np.array(img) / 255.0
    return img, arr


def add_label(image, text):
    labeled = image.convert("RGB")
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((0, 0, labeled.width, 20), fill=(0, 0, 0))
    draw.text((6, 3), text, fill=(255, 255, 255))
    return labeled


def make_grid(items, cols=2, cell_size=(240, 240), padding=12):
    rows = (len(items) + cols - 1) // cols
    width = cols * cell_size[0] + (cols + 1) * padding
    height = rows * cell_size[1] + (rows + 1) * padding
    canvas = Image.new("RGB", (width, height), (25, 25, 25))

    for idx, (title, image) in enumerate(items):
        row = idx // cols
        col = idx % cols
        x = padding + col * (cell_size[0] + padding)
        y = padding + row * (cell_size[1] + padding)
        tile = image.convert("RGB").resize(cell_size, Image.Resampling.NEAREST)
        tile = add_label(tile, title)
        canvas.paste(tile, (x, y))

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize CatCNN activations and weights")
    parser.add_argument("image_path", type=str, help="Path to an input image")
    parser.add_argument("--model-path", type=str, default="final_cnn_model.npz", help="Saved model weights file")
    parser.add_argument("--output-dir", type=str, default="visualizations", help="Directory for generated images")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cnn = CatCNN()
    load_model(cnn, args.model_path)

    original_img, x = load_input_image(args.image_path)
    prediction = float(np.squeeze(cnn.forward(x)))
    predicted_label = "cat" if prediction >= 0.5 else "not-cat"

    save_heatmap(x, output_dir / "input.png", resize_to=(256, 256))
    save_heatmap(cnn.conv1, output_dir / "conv1.png", resize_to=(256, 256))
    save_heatmap(cnn.pool1, output_dir / "pool1.png", resize_to=(256, 256))
    save_heatmap(cnn.conv2, output_dir / "conv2.png", resize_to=(256, 256))
    save_heatmap(cnn.pool2, output_dir / "pool2.png", resize_to=(256, 256))

    dense_weight_map = cnn.dense.W.reshape(14, 14)
    save_heatmap(dense_weight_map, output_dir / "dense_weights.png", resize_to=(256, 256))

    summary = Image.new("RGB", (800, 120), (35, 35, 35))
    draw = ImageDraw.Draw(summary)
    draw.text((16, 16), f"Prediction: {predicted_label}", fill=(255, 255, 255))
    draw.text((16, 44), f"Probability(cat): {prediction:.4f}", fill=(255, 255, 255))
    draw.text((16, 72), f"Model: {args.model_path}", fill=(200, 200, 200))
    summary.save(output_dir / "summary.png")

    grid = make_grid(
        [
            ("input", original_img),
            ("conv1", Image.open(output_dir / "conv1.png")),
            ("pool1", Image.open(output_dir / "pool1.png")),
            ("conv2", Image.open(output_dir / "conv2.png")),
            ("pool2", Image.open(output_dir / "pool2.png")),
            ("dense weights", Image.open(output_dir / "dense_weights.png")),
        ],
        cols=2,
        cell_size=(300, 300),
    )
    grid.save(output_dir / "feature_grid.png")

    print(f"Saved visualizations to {output_dir}")
    print(f"Prediction: {predicted_label} ({prediction:.4f})")


if __name__ == "__main__":
    main()
