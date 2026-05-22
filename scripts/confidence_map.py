import cv2
import numpy as np


def depth_jpg_to_confidence_tiff_old(jpg_path, tiff_path, k=0.1):
    depth = cv2.imread(jpg_path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise ValueError("Could not load image")

    if len(depth.shape) == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    depth = depth.astype(np.float32)

    if depth.max() > 0:
        depth = depth / depth.max()

    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)

    confidence = np.exp(-k * grad)
    confidence[depth == 0] = 0

    min_val = confidence.min()
    max_val = confidence.max()

    if max_val > min_val:
        confidence = (confidence - min_val) / (max_val - min_val)

    confidence_16 = (confidence * 65535).astype(np.uint16)

    cv2.imwrite(tiff_path, confidence_16)

    return confidence_16



def depth_jpg_to_confidence_tiff(jpg_path, tiff_path, k=10.0):
    depth = cv2.imread(jpg_path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise ValueError("Could not load image")

    if len(depth.shape) == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    depth = depth.astype(np.float32)

    depth = cv2.GaussianBlur(depth, (5, 5), 0)

    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)

    grad = grad / (grad.max() + 1e-6)

    confidence = np.exp(-k * grad)
    confidence[depth == 0] = 0
    confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-6)
    confidence_16 = (confidence * 65535).astype(np.uint16)

    cv2.imwrite(tiff_path, confidence_16)

    return confidence_16


if __name__ == '__main__':
    conf = depth_jpg_to_confidence_tiff(
        './depth_image.jpg',
        './test.tiff',
    )

    depth_map = np.zeros((480, 640))
    cv2.imwrite('./test2.tiff', depth_map)