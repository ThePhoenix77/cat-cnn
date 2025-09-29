import os
import numpy as np
from PIL import Image

def load_data(folder_cat, folder_noncat, img_size=(64, 64)):
    X, y = [], []
    for filename in os.listdir(folder_cat):
        img = Image.open(os.path.join(folder_cat, filename)).convert("L")
        img = img.resize(img_size)
        X.append(np.array(img) / 255.0)
        y.append(1)                                                             #cat label
    
    for filename in os.listdir(folder_noncat):
        img = Image.open(os.path.join(folder_noncat, filename)).convert("L")
        img = img.resize(img_size)
        X.append(np.array(img) / 255.0)
        y.append(0)                                                             #non-cat label
    
    return np.array(x), np.array(y)