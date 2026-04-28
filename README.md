# Cat-CNN: A Convolutional Neural Network for Cat Identification

<img width="1189" height="663" alt="cat-cnn-neuron-view" src="https://github.com/user-attachments/assets/fb6b060b-4a97-4fcb-8e01-1e7617621a90" />

A convolutional neural network for binary "cat vs not-cat" classification.

This repository is my playground where I built a CNN from scratch (NumPy + Pillow) without relying on any deep learning framework. Iterating on data loading, training, evaluation and small UX utilities. It started as exercises from Andrew Ng's deep learning materials plus some independent reading/tutorials, and I iterated on it with a bit of automated help.

Expect this project to be iteratively improved as it's intentionally simple and experimental for now.

## How It Works

The CNN processes a 64x64 image and passes it through the following architecture:

    input (64×64 grayscale) -> conv 3×3 (kernel1) -> conv map 62×62 -> max-pool 2×2 (stride 2) -> 31×31 ↩
                            -> conv 3×3 (kernel2) -> 29×29 -> max-pool 2×2 -> 14×14 -> flatten (196) ↩
                            -> Dense (196 -> 1) -> sigmoid -> output (probability)

## Project Overview

This project is a lightweight educational implementation of a CNN for binary image classification.

The model is intentionally small:

- two convolution layers
- two max-pooling layers
- one dense output layer
- sigmoid output for probability scoring

The goal is not state-of-the-art accuracy. The goal is to show how a CNN can be implemented end-to-end using only NumPy for math and Pillow for image handling.

## Example outputs

| Input | Conv1 |
|---|---|
| <img width="256" height="256" alt="input" src="https://github.com/user-attachments/assets/e2c88dab-898d-4f7d-953c-c82d2d1a3cdc" /> | <img width="256" height="256" alt="conv1" src="https://github.com/user-attachments/assets/f464a40f-352a-48c4-b68c-f3ce6eb7f9be" /> |

| Pool1 | Conv2 |
|---|---|
| <img width="256" height="256" alt="pool1" src="https://github.com/user-attachments/assets/d3afcc45-e21c-4fbd-ae2e-f312f8ea635e" /> | <img width="256" height="256" alt="conv2" src="https://github.com/user-attachments/assets/cb2d964e-2e0a-4eb6-9631-c8d82f808b59" /> |

| Pool2 | Dense weights |
|---|---|
| <img width="256" height="256" alt="pool2" src="https://github.com/user-attachments/assets/baed1b79-1329-4f98-8e9b-97fbd1306e1c" /> | <img width="256" height="256" alt="dense_weights" src="https://github.com/user-attachments/assets/bfa32d93-e94a-40d4-9551-78be7b53db42" /> |

| Feature grid |
|---|
| <img width="636" height="948" alt="feature_grid" src="https://github.com/user-attachments/assets/9a2f449e-6905-44e2-a592-056dfee9d9d2" /> |

| Summary |
|---|
| <img width="800" height="120" alt="summary" src="https://github.com/user-attachments/assets/770dec69-dc0b-45b0-87cd-c9e3030040db" /> |

## CNN Concept

Briefly, a convolutional neural network works by learning filters that respond to local patterns in an image.

### Convolution

The 3×3 kernels slide across the image and compute dot products with small neighborhoods. Early filters tend to learn simple patterns such as:

- edges
- corners
- light/dark transitions
- simple textures

In this project, the first convolution layer extracts local image patterns from the 64×64 grayscale input. The second convolution layer operates on pooled features and can combine earlier patterns into slightly more abstract ones.

### Pooling

Max-pooling reduces spatial size by keeping the strongest activation in each small region. This helps:

- reduce computation
- reduce sensitivity to tiny shifts
- keep the most salient features

### Flatten + Dense + Sigmoid

After convolution and pooling, the feature map is flattened into a 1D vector. A dense layer maps that vector to a single output logit, and sigmoid turns that into a value between 0 and 1.

Interpretation:

- close to 1.0 → cat
- close to 0.0 → not cat

## Repository Structure

- `cat_cnn.py` — CNN forward and backward pass
- `cnn_layers.py` — convolution and max-pooling operations
- `cnn_network.py` — dense layer and gradient descent optimizer
- `data_loader.py` — folder-based image loading
- `main.py` — training entrypoint, prediction loop, model save/load, metrics export
- `prepare_kaggle_data.py` — converts Kaggle raw dataset files into `data/cats` and `data/noncats`
- `utils.py` — sigmoid, loss, and helper functions
- `requirements.txt` — minimal runtime dependencies

## Requirements

Core runtime dependencies:

- Python 3.10+
- NumPy
- Pillow

Install the base dependencies:

```bash
pip install -r requirements.txt
```

The requirements file now includes the Kaggle workflow helpers as well, so the command above covers the full project setup.

If you want to install only the core runtime dependencies, use:

```bash
pip install numpy Pillow
```

## Dataset Format

The training script expects a folder layout like this:

```text
data/
├── cats/
└── noncats/
```

The loader:

- reads images from both folders
- converts them to grayscale
- resizes them to 64×64
- normalizes pixel values to the range [0, 1]

## Kaggle Dataset Workflow

In my model training I used `sagar2522/cat-vs-non-cat` Kaggle dataset that you can find [here](https://www.kaggle.com/datasets/sagar2522/cat-vs-non-cat).

### 1) Add Kaggle credentials

Use either:

- `kaggle/kaggle.json` in the repo, or
- `~/.kaggle/kaggle.json`

Example file:

```json
{
  "username": "YOUR_USERNAME",
  "key": "YOUR_API_KEY"
}
```

If you use `~/.kaggle/kaggle.json`, set strict permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### 2) Download the dataset

```bash
KAGGLE_CONFIG_DIR="$PWD/kaggle" kaggle datasets download -d sagar2522/cat-vs-non-cat -p data_raw
unzip -o data_raw/cat-vs-non-cat.zip -d data_raw
```

### 3) Convert raw files into training folders

```bash
python prepare_kaggle_data.py --source-root data_raw --output-root data --max-per-class 300
```

This generates:

- `data/cats`
- `data/noncats`

## Training

Train the model and save the weights:

```bash
python main.py --train-only --epochs 10 --lr 0.01 --save-path final_cnn_model.npz --metrics-path training_metrics.json
```

What happens during training:

1. load training images from `data/cats` and `data/noncats`
2. run the CNN forward pass
3. compute binary cross-entropy loss
4. backpropagate gradients
5. update kernels and dense layer parameters with gradient descent
6. save the model to `final_cnn_model.npz`
7. evaluate metrics and save them to `training_metrics.json`

## Metrics

Training metrics are computed on the loaded training set after the final epoch.

They are printed to the console and saved as JSON.

Fields in the metrics report:

- `loss`
- `accuracy`
- `precision`
- `recall`
- `f1`
- `tp`
- `tn`
- `fp`
- `fn`
- `samples`

### Latest Run

Based on the current `training_metrics.json`:

| Metric | Value |
|---|---:|
| Loss | 0.6811 |
| Accuracy | 0.6555 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| TP | 0 |
| TN | 137 |
| FP | 0 |
| FN | 72 |
| Samples | 209 |

## Prediction Mode

If you run `main.py` without `--train-only`, the script trains and then enters an interactive prediction loop:

```bash
python main.py
```

You will be prompted for an image path. The image will be resized to 64×64, converted to grayscale, and classified as:

- `Cat`
- `Not a cat`

Press `Ctrl+C` or send EOF to exit.

## Visualizing the Network

You can generate an activation and weight-map images for a single prediction just like the images presented at the top:

```bash
python visualize_model.py path/to/image.jpg --model-path final_cnn_model.npz --output-dir visualizations
```

Quick example using one of the prepared training images:

```bash
SAMPLE=$(find data/cats -type f | head -n 1)
python visualize_model.py "$SAMPLE" --model-path final_cnn_model.npz --output-dir visualizations
open visualizations
```

This creates PNG files showing:

- the input image
- `conv1` activations
- `pool1` activations
- `conv2` activations
- `pool2` activations
- the dense-layer weight map reshaped to the pooled feature size
- a summary image with the predicted label and probability

The output is saved into the chosen directory and can be opened like normal images.

## Animated Neuron Network(Abstract)

You can also generate an abstract animated neuron view that uses actual values from your trained model prediction.

<img width="1262" height="929" alt="neuron-network-visualizer" src="https://github.com/user-attachments/assets/448a6234-be7d-4f2a-a7e4-0fc03500c06e" />

### 1) Export activation data from a real prediction

```bash
SAMPLE=$(find data/cats -type f | head -n 1)
python export_animation_data.py "$SAMPLE" --model-path final_cnn_model.npz --output-json visualizations/network_animation_data.json
```

### 2) Open the animation page

Open the file `network_animation.html` in your browser.

- It will try to load `visualizations/network_animation_data.json` by default.
- If blocked by browser file permissions, run a local server:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/network_animation.html`.

The page animates each layer as abstract neurons where:

- node size/glow tracks activation magnitude
- color encodes sign (positive/negative)
- animated links indicate signal flow between layers
- output panel shows final prediction and probability

## CLI Reference

### `main.py`

- `--epochs` — number of training epochs, default `10`
- `--lr` — learning rate, default `0.01`
- `--data-cats` — path to cat images, default `data/cats`
- `--data-noncats` — path to non-cat images, default `data/noncats`
- `--save-path` — output model path, default `final_cnn_model.npz`
- `--metrics-path` — output metrics JSON path, default `training_metrics.json`
- `--load-model` — load an existing `.npz` model before training
- `--train-only` — train, save, evaluate, and exit without interactive prediction

### `prepare_kaggle_data.py`

- `--source-root` — root folder containing extracted Kaggle files, default `data_raw`
- `--output-root` — output dataset root, default `data`
- `--max-per-class` — cap images per class, default `300`
- `--seed` — random seed, default `42`

### `visualize_model.py`

- `image_path` — image to inspect
- `--model-path` — saved `.npz` model file, default `final_cnn_model.npz`
- `--output-dir` — folder to write PNG visualizations, default `visualizations`

### `export_animation_data.py`

- `image_path` — image to inspect for animation data
- `--model-path` — saved `.npz` model file, default `final_cnn_model.npz`
- `--output-json` — output JSON data for animation, default `visualizations/network_animation_data.json`

## Saved Model Format

The saved `.npz` file stores the learned parameters:

- `kernel1`
- `kernel2`
- `dense_W`
- `dense_b`

## Implementation Notes

- Convolution and pooling are implemented manually in Python/Numpy.
- Backpropagation is also implemented manually for learning purposes.
- `sigmoid` and other math helpers are vectorized with NumPy.
- The `visualize_model.py` and the `network_animation.html` are the only files implemented using ai assisted synthesis(vibe-coded).
- The project is small enough to understand end-to-end, but it is not optimized for performance.

## Current Status

The repo already includes a trained model artifact and metrics output from a recent run:

- [final_cnn_model.npz](final_cnn_model.npz)
- [training_metrics.json](training_metrics.json)



## License

See [LICENSE](LICENSE) for details.
