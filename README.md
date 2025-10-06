# Cat-CNN: A Convolutional Neural Network for Cat Identification

This project is a lightweight, educational implementation of a convolutional neural network (CNN) designed to identify images of cats. It is written in Python and uses NumPy for all numerical operations, providing a from-scratch look at how a CNN works without relying on high-level deep learning frameworks.

## Features

- **Minimalist Implementation**: Built from the ground up using only NumPy for core computations.
- **Two-Layer CNN**: A simple yet effective architecture with two convolutional layers, each followed by max-pooling.
- **Gradient-Based Optimization**: Uses gradient descent to train the model and minimize loss.
- **Easy to Use**: Includes a straightforward command-line interface for training and prediction.

## How It Works

The CNN processes a 64x64 grayscale image and passes it through the following layers:

1.  **Convolutional Layer 1**: Applies a 3x3 kernel to detect low-level features like edges and textures.
2.  **Max-Pooling Layer 1**: Downsamples the feature map to reduce dimensionality.
3.  **Convolutional Layer 2**: Applies another 3x3 kernel to learn more complex features.
4.  **Max-Pooling Layer 2**: Further downsamples the feature map.
5.  **Flatten Layer**: Converts the 2D feature map into a 1D vector.
6.  **Dense Layer**: A fully connected layer that produces the final output.
7.  **Sigmoid Activation**: Outputs a probability score between 0 and 1, indicating whether the image is a cat.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy
- Pillow

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/cat-cnn.git
    cd cat-cnn
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Dataset

This project requires a dataset of cat and non-cat images. Due to the educational nature of this implementation, a small, curated dataset is recommended for efficient training.

1.  Create the following directory structure:
    ```
    data/
    ├── cats/
    └── noncats/
    ```

2.  Place your cat images in the `data/cats` folder and non-cat images in the `data/noncats` folder. The model is designed to work with 64x64 grayscale images, but it will automatically convert and resize them as needed.

## Usage

To train the model, run the `main.py` script:

```bash
python main.py
```

The script will first load the data, then begin the training process, printing the loss at each epoch. After training is complete, you will be prompted to enter the path to an image for prediction.

### Example

```
Loading data...
Training CNN...
Epoch 1, Loss: 0.693
Epoch 2, Loss: 0.685
...
Epoch 10, Loss: 0.532

Enter image path: /path/to/your/image.jpg
Cat
```

To exit the prediction loop, press `Ctrl+C`.