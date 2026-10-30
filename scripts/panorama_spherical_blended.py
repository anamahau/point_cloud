#!/usr/bin/env python3
"""ROS Noetic: reconstruct a calibrated spherical panorama with multiband blending.

Uses the existing spherical_calibration.json and spherical_maps.npz files.
No image registration or new calibration is performed.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import rospy


def pyramid_blend(images, weights, levels):
    """Laplacian pyramid blending with normalized Gaussian weight pyramids."""
    height, width = images[0].shape[:2]
    max_levels = max(1, int(np.floor(np.log2(max(2, min(height, width))))) - 2)
    levels = max(1, min(int(levels), max_levels))

    numerator = None
    denominator = None
    for image, weight in zip(images, weights):
        gp_img = [image.astype(np.float32)]
        gp_weight = [weight.astype(np.float32)]
        for _ in range(levels):
            gp_img.append(cv2.pyrDown(gp_img[-1]))
            gp_weight.append(cv2.pyrDown(gp_weight[-1]))

        lap = []
        for level in range(levels):
            size = (gp_img[level].shape[1], gp_img[level].shape[0])
            lap.append(gp_img[level] - cv2.pyrUp(gp_img[level + 1], dstsize=size))
        lap.append(gp_img[-1])

        if numerator is None:
            numerator = [np.zeros_like(layer) for layer in lap]
            denominator = [np.zeros_like(w) for w in gp_weight]
        for level in range(levels + 1):
            numerator[level] += lap[level] * gp_weight[level][:, :, None]
            denominator[level] += gp_weight[level]

    blended = [num / np.maximum(den, 1e-8)[:, :, None]
               for num, den in zip(numerator, denominator)]
    result = blended[-1]
    for level in range(levels - 1, -1, -1):
        size = (blended[level].shape[1], blended[level].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + blended[level]
    return result


def build_panorama(images, calibration_dir, feather_px=100.0, levels=5):
    with (calibration_dir / 'spherical_calibration.json').open() as f:
        meta = json.load(f)
    width, height = map(int, meta['size'])
    expected = tuple(reversed(meta['input_size']))
    if len(images) != 6 or any(im is None or im.shape[:2] != expected for im in images):
        raise ValueError('Expected six BGR images of size %s' % (meta['input_size'],))

    warped_images, weights, masks = [], [], []
    with np.load(str(calibration_dir / 'spherical_maps.npz')) as maps:
        for i, image in enumerate(images, 1):
            mx = maps['mapx_%d' % i]
            my = maps['mapy_%d' % i]
            if mx.shape != (height, width) or my.shape != (height, width):
                raise ValueError('Invalid map shape for image %d: %s' % (i, mx.shape))
            warped = cv2.remap(image, mx, my, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
            # Validity is determined from remapping a full source mask, not pixel color.
            src_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            mask = cv2.remap(src_mask, mx, my, interpolation=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT) > 0
            dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            # Keep a nonzero contribution at the valid edge, avoiding tiny holes.
            weight = np.where(mask, np.minimum((dist + 1.0) / max(feather_px, 1.0), 1.0), 0.0)
            warped_images.append(warped)
            weights.append(weight.astype(np.float32))
            masks.append(mask)
            rospy.loginfo('Image %d: %d valid panorama pixels', i, int(mask.sum()))

    output = pyramid_blend(warped_images, weights, levels)
    # Restore true black outside the union of valid warped images.
    union = np.logical_or.reduce(masks)
    output = np.clip(output, 0, 255).astype(np.uint8)
    output[~union] = 0
    return output


def main():
    rospy.init_node('panorama_spherical_blended')
    folder = Path(rospy.get_param('~input_dir', '/talos_ws/samples/sample_0075'))
    cal = Path(rospy.get_param('~calibration_dir', '/talos_ws/samples/spherical_calibration'))
    feather = float(rospy.get_param('~feather_px', 100.0))
    levels = int(rospy.get_param('~blend_levels', 5))
    sid = folder.name.replace('sample_', '', 1)
    paths = [folder / ('image_%s_%d.png' % (sid, i)) for i in range(1, 7)]
    images = [cv2.imread(str(path)) for path in paths]
    for path, image in zip(paths, images):
        if image is None:
            raise RuntimeError('Cannot read image: %s' % path)
    pano = build_panorama(images, cal, feather, levels)
    output = folder / ('panorama_%s_spherical_blended.png' % sid)
    if not cv2.imwrite(str(output), pano):
        raise RuntimeError('Failed to save %s' % output)
    rospy.loginfo('Saved blended panorama: %s', output)


if __name__ == '__main__':
    main()
