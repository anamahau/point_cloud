#!/usr/bin/env python3
"""ROS 1 Noetic: reconstruct panorama from six fixed image-to-panorama homographies."""
from pathlib import Path
import cv2
import numpy as np
import rospy

WIDTH = 1066
HEIGHT = 720
HOMOGRAPHIES = [
    [[1.2308189339185382, 0.17522150458379745, -61.39598524757333], [0.05842527164075883, 1.249725289847907, -36.031737370541435], [0.0001748035225252821, 0.00037269393780084057, 1.0]],
    [[1.330693100976843, 0.13684688973519812, 166.81216036307364], [0.11708969698625646, 1.2766641574816273, -66.35046406020685], [0.0002396234269531617, 0.0003203073535154418, 1.0]],
    [[0.9305054612301141, 0.05576940106806328, 452.7026233552056], [0.04561881658926967, 1.0291008158981214, -8.343351328401964], [-7.552885739406266e-05, 0.00017395749756470114, 1.0]],
    [[0.47760919937416324, -0.1299940620581446, 482.2497254781987], [-0.2140144690757459, 0.574095807967057, 367.81622922243963], [-0.0003308366048579736, -0.0002274345037445178, 1.0]],
    [[0.9471603141288972, -0.22112623230381107, 274.7649392647822], [0.06531285613809548, 0.5787433557366003, 353.5916617303706], [0.0001987309874833894, -0.00043534502739493535, 1.0]],
    [[1.356558529542893, -0.01835335197390868, -14.76586160742802], [0.30761121532052543, 0.9055570899312149, 314.7963728623251], [0.0006122317566944508, -0.00015481275026224146, 1.0]],
]


def feather_weights(height, width, feather_px):
    """Smooth source-edge weights, excluding pixels outside source bounds."""
    mask = np.full((height, width), 255, dtype=np.uint8)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    weights = np.minimum(distance / max(float(feather_px), 1.0), 1.0)
    return weights.astype(np.float32)


def main():
    rospy.init_node('quality_control_panorama_fixed')
    input_dir = Path(rospy.get_param('~input_dir', '/talos_ws/sample_0075'))
    feather_px = float(rospy.get_param('~feather_px', 35.0))
    output_name = rospy.get_param('~output_name', 'panorama_0075_fixed.png')
    if feather_px <= 0:
        raise ValueError('feather_px must be positive')

    numerator = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    denominator = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    for idx, matrix in enumerate(HOMOGRAPHIES, start=1):
        image_path = input_dir / ('image_0075_%d.png' % idx)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('Cannot read image: %s' % image_path)
        if image.shape[:2] != (480, 640):
            raise ValueError('Expected 640x480 image: %s' % image_path)

        H = np.asarray(matrix, dtype=np.float64)
        warped = cv2.warpPerspective(image, H, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
        weights = feather_weights(image.shape[0], image.shape[1], feather_px)
        warped_weights = cv2.warpPerspective(weights, H, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
        numerator += warped.astype(np.float32) * warped_weights[:, :, None]
        denominator += warped_weights
        rospy.loginfo('Warped %s', image_path.name)

    panorama = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    valid = denominator > 1e-6
    panorama[valid] = np.clip(
        numerator[valid] / denominator[valid, None], 0, 255
    ).astype(np.uint8)

    output_path = input_dir / output_name
    if not cv2.imwrite(str(output_path), panorama):
        raise RuntimeError('Failed to write panorama: %s' % output_path)
    rospy.loginfo('Saved panorama: %s (%dx%d)', output_path, WIDTH, HEIGHT)
    rospy.loginfo('Covered pixels: %.1f%%', 100.0 * np.mean(valid))


if __name__ == '__main__':
    try:
        main()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
    except Exception as exc:
        rospy.logerr('Panorama reconstruction failed: %s', exc)
        raise
