import os
import numpy as np
from PIL import Image

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
NORMAL_DIR = os.path.join(BASE_DIR, "normal")
ABNORMAL_DIR = os.path.join(BASE_DIR, "abnormal")

def generate_images(num_images=20, size=(32, 32)):
    os.makedirs(NORMAL_DIR, exist_ok=True)
    os.makedirs(ABNORMAL_DIR, exist_ok=True)
    for i in range(num_images):
        normal_array = np.random.randint(0, 50, (size[0], size[1], 3), dtype=np.uint8)
        normal_array[..., 2] = np.random.randint(150, 255, (size[0], size[1]))
        Image.fromarray(normal_array).save(os.path.join(NORMAL_DIR, f"normal_{i}.png"))
        abnormal_array = np.random.randint(0, 50, (size[0], size[1], 3), dtype=np.uint8)
        abnormal_array[..., 0] = np.random.randint(150, 255, (size[0], size[1]))
        Image.fromarray(abnormal_array).save(os.path.join(ABNORMAL_DIR, f"abnormal_{i}.png"))
    print(f"✅ Generated {num_images} dummy normal and abnormal images each.")

if __name__ == "__main__":
    generate_images()
