from PIL import Image
import numpy as np

def load_image(uploaded):
    return Image.open(uploaded).convert("RGB")

def pil_to_numpy(img):
    return np.array(img)
