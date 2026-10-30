#!/usr/bin/env python3
"""ROS Noetic: reconstruct panorama with calibrated spherical inverse maps."""
import json
from pathlib import Path
import cv2
import numpy as np
import rospy


def build_panorama(images, calibration_dir, feather_px=35.0):
    with (calibration_dir / 'spherical_calibration.json').open() as f:
        meta = json.load(f)
    W, H = meta['size']
    expected = tuple(reversed(meta['input_size']))
    if len(images) != 6 or any(im is None or im.shape[:2] != expected for im in images):
        raise ValueError('Expected six BGR images of size %s' % (meta['input_size'],))
    accum = np.zeros((H, W, 3), np.float32)
    total = np.zeros((H, W), np.float32)
    with np.load(str(calibration_dir / 'spherical_maps.npz')) as maps:
        for i, image in enumerate(images, 1):
            mx, my = maps['mapx_%d' % i], maps['mapy_%d' % i]
            warped = cv2.remap(image, mx, my, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
            source_mask = np.full(image.shape[:2], 255, np.uint8)
            mask = cv2.remap(source_mask, mx, my, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT) > 0
            distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
            weight = np.minimum(distance / max(feather_px, 1.0), 1.0)
            accum += warped.astype(np.float32) * weight[:, :, None]
            total += weight
    pano = np.zeros((H, W, 3), np.uint8)
    valid = total > 0
    pano[valid] = np.clip(accum[valid] / total[valid, None], 0, 255).astype(np.uint8)
    return pano


def main():
    rospy.init_node('panorama_spherical_runtime')
    folder = Path(rospy.get_param('~input_dir', '/talos_ws/sample_0075'))
    cal = Path(rospy.get_param('~calibration_dir', '/talos_ws/sample_0075/spherical_calibration'))
    feather = float(rospy.get_param('~feather_px', 35.0))
    sid = folder.name.replace('sample_', '', 1)
    files = [folder / ('image_%s_%d.png' % (sid, i)) for i in range(1, 7)]
    images = [cv2.imread(str(p)) for p in files]
    pano = build_panorama(images, cal, feather)
    output = folder / ('panorama_%s_spherical.png' % sid)
    if not cv2.imwrite(str(output), pano):
        raise RuntimeError('Failed to write %s' % output)
    rospy.loginfo('Saved panorama: %s', output)

if __name__ == '__main__':
    main()
