#!/usr/bin/env python3
"""ROS Noetic: warp six images into a fixed 1066x720 panorama using saved H only.

No feature detection, matching, optimization or cv2.Stitcher is performed.
The H matrices map original 640x480 image pixels to panorama pixels.
"""
from pathlib import Path
import re
import cv2
import numpy as np
import rospy

SIZE = (1066, 720)  # width, height
SOURCE_SIZE = (640, 480)
HOMOGRAPHIES = np.asarray([
    [[1.2308189339185382, 0.17522150458379745, -61.39598524757333], [0.05842527164075883, 1.249725289847907, -36.031737370541435], [0.0001748035225252821, 0.00037269393780084057, 1.0]],
    [[1.330693100976843, 0.13684688973519812, 166.81216036307364], [0.11708969698625646, 1.2766641574816273, -66.35046406020685], [0.0002396234269531617, 0.0003203073535154418, 1.0]],
    [[0.9305054612301141, 0.05576940106806328, 452.7026233552056], [0.04561881658926967, 1.0291008158981214, -8.343351328401964], [-7.552885739406266e-05, 0.00017395749756470114, 1.0]],
    [[0.47760919937416324, -0.1299940620581446, 482.2497254781987], [-0.2140144690757459, 0.574095807967057, 367.81622922243963], [-0.0003308366048579736, -0.0002274345037445178, 1.0]],
    [[0.9471603141288972, -0.22112623230381107, 274.7649392647822], [0.06531285613809548, 0.5787433557366003, 353.5916617303706], [0.0001987309874833894, -0.00043534502739493535, 1.0]],
    [[1.356558529542893, -0.01835335197390868, -14.76586160742802], [0.30761121532052543, 0.9055570899312149, 314.7963728623251], [0.0006122317566944508, -0.00015481275026224146, 1.0]],
], dtype=np.float64)


def main():
    rospy.init_node('panorama_from_saved_homographies')
    folder = Path(rospy.get_param('~input_dir', '/talos_ws/sample_0075'))
    feather_px = float(rospy.get_param('~feather_px', 35.0))
    debug = bool(rospy.get_param('~save_warps', False))
    sample_match = re.fullmatch(r'sample_(\d+)', folder.name)
    if not sample_match:
        raise ValueError('Expected directory named sample_NNNN, got: %s' % folder)
    sample_id = sample_match.group(1)
    if feather_px <= 0:
        raise ValueError('feather_px must be > 0')

    accum = np.zeros((SIZE[1], SIZE[0], 3), np.float32)
    total = np.zeros((SIZE[1], SIZE[0]), np.float32)
    debug_dir = folder / 'warped_debug'
    if debug:
        debug_dir.mkdir(exist_ok=True)

    # Source-space feathering: same fixed geometry for every sample.
    source_mask = np.full((SOURCE_SIZE[1], SOURCE_SIZE[0]), 255, np.uint8)
    edge_dist = cv2.distanceTransform(source_mask, cv2.DIST_L2, 3)
    source_weight = np.minimum(edge_dist / feather_px, 1.0).astype(np.float32)

    for i, H in enumerate(HOMOGRAPHIES, start=1):
        path = folder / ('image_%s_%d.png' % (sample_id, i))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError('Cannot read: %s' % path)
        if (image.shape[1], image.shape[0]) != SOURCE_SIZE:
            raise ValueError('%s: expected 640x480, got %dx%d' %
                             (path, image.shape[1], image.shape[0]))

        warped = cv2.warpPerspective(image, H, SIZE, flags=cv2.INTER_LINEAR)
        weights = cv2.warpPerspective(source_weight, H, SIZE,
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)
        accum += warped.astype(np.float32) * weights[:, :, None]
        total += weights
        if debug:
            cv2.imwrite(str(debug_dir / ('warp_%d.png' % i)), warped)
        rospy.loginfo('Warped image %d/6', i)

    panorama = np.zeros_like(accum, dtype=np.uint8)
    valid = total > 1e-6
    panorama[valid] = np.clip(accum[valid] / total[valid, None], 0, 255).astype(np.uint8)
    out = folder / ('panorama_%s_homography.png' % sample_id)
    if not cv2.imwrite(str(out), panorama):
        raise IOError('Cannot write: %s' % out)
    rospy.loginfo('Saved %s; coverage %.1f%%', out, 100 * valid.mean())


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
