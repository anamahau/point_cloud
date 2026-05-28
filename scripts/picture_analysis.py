#!/usr/bin/env python3

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':

    print(':)')

    img = Image.open('./image_left.png')
    
    arr = np.array(img)
    print(arr[:, 0, :])
    print(arr[:, 0, :].shape)
    print(arr.shape)

    black_rows = np.all(arr == [0, 0, 0], axis=(1, 2))
    count = np.sum(black_rows)
    print(count)

'''
So you need:
- current width = 640
- target width = 768
- padding needed = 128 pixels total
- 64 pixels on each side

    from PIL import Image
    # Load image
    img = Image.open("image.png")
    width, height = img.size
    # Target 16:9 width
    target_width = int(height * 16 / 9)
    # Create black canvas
    canvas = Image.new("RGB", (target_width, height), (0, 0, 0))
    # Center original image
    x_offset = (target_width - width) // 2
    canvas.paste(img, (x_offset, 0))
    # Save result
    canvas.save("image_16_9.png")
    print(canvas.size)
'''