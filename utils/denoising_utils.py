import os
from .common_utils import *


# 对numpy格式的img加噪函数，噪声程度为sigma     
# 返回值是噪声图像的PIL和numpy的两种格式   

def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """

    # 加入高斯噪声/possion噪声
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    #img_noisy_np = np.clip(img_np + np.random.poisson(sigma, size=img_np.shape), 0, 1).astype(np.float32) # lam>=0 值越小，噪声频率就越少

    img_noisy_pil = np_to_pil(img_noisy_np)

    # 返回两种格式的加噪后的图像(pil,np)
    return img_noisy_pil, img_noisy_np

