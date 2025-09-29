import numpy as np

def conv2d(image, kernel):
    h, w = image.shape
    k = kernel.shape[0]
    out_h, out_w = h - k + 1, w - k + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = image[i:i+k, j:j+k]
            output[i, j] = np.sum(region * kernel)
    return output

def max_pooling(image, size=2, stride=2):
    h, w = image.shape
    out_h, out_w = (h - size)//stride + 1, (w - size)//stride + 1
    output = np.zeros((out_h, out_w))
    for i in range(0, h - size + 1, stride):
        for j in range(0, w - size + 1, stride):
            region = image[i:i+size, j:j+size]
            output[i//stride, j//stride] = np.max(region)
    return output

def conv2d_backward(dL_dy, x, kernel):
    k_h, k_w = kernel.shape
    i_h, i_w = x.shape
    dL_dk = np.zeros_like(kernel)
    for m in range(k_h):
        for n in range(k_w):
            dL_dk[m, n] = np.sum(dL_dy * x[m:m+dL_dy.shape[0], n:n+dL_dy.shape[1]])
    pad_h, pad_w = k_h-1, k_w-1
    padded_dL_dy = np.pad(dL_dy, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    flipped_kernel = np.flip(kernel)
    dL_dx = conv2d(padded_dL_dy, flipped_kernel)
    return dL_dx, dL_dk

def max_pool_backward(dL_dy, x, size=2, stride=2):
    h, w = x.shape
    out_h, out_w = dL_dy.shape
    dL_dx = np.zeros_like(x)
    for i in range(out_h):
        for j in range(out_w):
            region = x[i*stride:i*stride+size, j*stride:j*stride+size]
            max_val = np.max(region)
            for m in range(size):
                for n in range(size):
                    if region[m, n] == max_val:
                        dL_dx[i*stride+m, j*stride+n] = dL_dy[i, j]
                        break
    return dL_dx
