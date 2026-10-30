#!/usr/bin/env python3
"""ROS Noetic: calibrate fixed spherical maps from six images.
Requires OpenCV Python bindings with cv2.detail stitching APIs.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import rospy

N = 6

def images_from_folder(folder):
    files = [folder / ('image_%s_%d.png' % (folder.name.replace('sample_', ''), i)) for i in range(1, N + 1)]
    imgs = [cv2.imread(str(p)) for p in files]
    if any(x is None for x in imgs):
        raise RuntimeError('Missing/unreadable images: %s' % files)
    if len(set((im.shape[1], im.shape[0]) for im in imgs)) != 1:
        raise RuntimeError('All images must have equal dimensions')
    return imgs

def calibrate(imgs, outdir):
    cv2.setRNGSeed(42)
    finder = cv2.SIFT_create(nfeatures=8000)
    features = [cv2.detail.computeImageFeatures2(finder, im) for im in imgs]
    matcher = cv2.detail.BestOf2NearestMatcher_create(False, 0.3)
    pairwise = matcher.apply2(features)
    matcher.collectGarbage()
    estimator = cv2.detail.HomographyBasedEstimator()
    ok, cameras = estimator.apply(features, pairwise, None)
    if not ok:
        raise RuntimeError('Camera estimation failed')
    for c in cameras:
        c.R = c.R.astype(np.float32)
    adjuster = cv2.detail.BundleAdjusterRay()
    adjuster.setConfThresh(0.3)
    refine = np.zeros((3, 3), np.uint8)
    refine[0, 0] = refine[0, 1] = refine[0, 2] = refine[1, 1] = refine[1, 2] = 1
    adjuster.setRefinementMask(refine)
    ok, cameras = adjuster.apply(features, pairwise, cameras)
    if not ok:
        raise RuntimeError('Bundle adjustment failed')
    rotations = cv2.detail.waveCorrect([c.R for c in cameras], cv2.detail.WAVE_CORRECT_HORIZ)
    focals = sorted(float(c.focal) for c in cameras)
    scale = float(np.median(focals))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError('Invalid warping scale')
    warper = cv2.PyRotationWarper('spherical', scale)
    rois = []
    maps = []
    for im, c, R in zip(imgs, cameras, rotations):
        K = c.K().astype(np.float32)
        R = np.asarray(R, np.float32)
        # roi, mx, my = warper.buildMaps((im.shape[1], im.shape[0]), K, R)
        # x, y, w, h = map(int, roi)
        # if mx.shape != (h, w):
        #     raise RuntimeError('Unexpected warper ROI/map dimensions: %s / %s' % (roi, mx.shape))
        # rois.append((x, y, w, h))
        # maps.append((mx, my))
        roi, mx, my = warper.buildMaps((im.shape[1], im.shape[0]), K, R)
        x, y, _, _ = map(int, roi)
        # Use actual dimensions of generated maps
        h, w = mx.shape[:2]
        if mx.shape != my.shape:
            raise RuntimeError(
                'Map dimensions do not match: %s / %s'
                % (mx.shape, my.shape)
            )
        rois.append((x, y, w, h))
        maps.append((mx, my))
    left = min(r[0] for r in rois)
    top = min(r[1] for r in rois)
    right = max(r[0] + r[2] for r in rois)
    bottom = max(r[1] + r[3] for r in rois)
    W, H = right-left, bottom-top
    if W * H > 80000000:
        raise RuntimeError('Panorama canvas unexpectedly large')
    metadata = {'projection': 'spherical', 'size': [W, H],
                'input_size': [imgs[0].shape[1], imgs[0].shape[0]],
                'warping_scale': scale, 'images': []}
    payload = {}
    for i, (roi, (mx, my)) in enumerate(zip(rois, maps), 1):
        x, y, w, h = roi
        # Full canvas inverse remap. -1 coordinates outside this image.
        fullx = np.full((H, W), -1, np.float32)
        fully = np.full((H, W), -1, np.float32)
        xx, yy = x-left, y-top
        fullx[yy:yy+h, xx:xx+w] = mx
        fully[yy:yy+h, xx:xx+w] = my
        payload['mapx_%d' % i] = fullx
        payload['mapy_%d' % i] = fully
        metadata['images'].append({'index': i, 'roi': list(roi)})
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(outdir / 'spherical_maps.npz'), **payload)
    with (outdir / 'spherical_calibration.json').open('w') as f:
        json.dump(metadata, f, indent=2)
    rospy.loginfo('Saved spherical maps: %s (%dx%d)', outdir, W, H)

if __name__ == '__main__':
    rospy.init_node('calibrate_spherical_panorama')
    folder = Path(rospy.get_param('~input_dir', '/talos_ws/sample_0075'))
    outdir = Path(rospy.get_param('~calibration_dir', str(folder / 'spherical_calibration')))
    calibrate(images_from_folder(folder), outdir)
