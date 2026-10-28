#!/usr/bin/env python3
"""ROS 1 Noetic: stitch six disk images and estimate image-to-panorama homographies."""
import json
from pathlib import Path
import cv2
import numpy as np
import rospy


class PanoramaNode:
    def __init__(self):
        self.input_dir = Path(rospy.get_param('~input_dir', 'sample_0075'))
        self.num_images = int(rospy.get_param('~num_images', 6))
        self.min_inliers = int(rospy.get_param('~min_inliers', 12))
        self.ratio = float(rospy.get_param('~ratio_threshold', 0.75))
        self.process()

    def load_images(self):
        paths = sorted(self.input_dir.glob('image_*.png'),
                       key=lambda p: int(p.stem.rsplit('_', 1)[-1]))
        if len(paths) != self.num_images:
            raise RuntimeError('Expected {} images; found {}'.format(self.num_images, len(paths)))
        images = [cv2.imread(str(p)) for p in paths]
        if any(im is None for im in images):
            raise RuntimeError('Could not read one or more images')
        return paths, images

    def make_panorama(self, images):
        for mode in (cv2.Stitcher_PANORAMA, cv2.Stitcher_SCANS):
            stitcher = cv2.Stitcher_create(mode)
            status, pano = stitcher.stitch(images)
            if status == cv2.Stitcher_OK and pano is not None:
                rospy.loginfo('Panorama created (mode %s), size %sx%s',
                              mode, pano.shape[1], pano.shape[0])
                return pano, ('PANORAMA' if mode == cv2.Stitcher_PANORAMA else 'SCANS')
            rospy.logwarn('Stitching mode %s failed: %s', mode, status)
        raise RuntimeError('Panorama stitching failed in both modes')

    def estimate_transform(self, image, panorama):
        """Independent approximation; NOT the Stitcher internal camera warp."""
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        detector = cv2.SIFT_create(nfeatures=8000)
        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return None, 0, 0
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        pairs = matcher.knnMatch(des1, des2, k=2)
        good = [m for pair in pairs if len(pair) == 2
                for m, n in [pair] if m.distance < self.ratio * n.distance]
        if len(good) < 4:
            return None, len(good), 0
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        inliers = int(mask.sum()) if mask is not None else 0
        if H is None or inliers < self.min_inliers or not np.isfinite(H).all():
            return None, len(good), inliers
        H = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H
        return H, len(good), inliers

    def process(self):
        paths, images = self.load_images()
        panorama, mode = self.make_panorama(images)
        sample_id = self.input_dir.name.replace('sample_', '', 1)
        panorama_path = self.input_dir / ('panorama_' + sample_id + '.png')
        json_path = self.input_dir / ('transformations_' + sample_id + '.json')
        if not cv2.imwrite(str(panorama_path), panorama):
            raise RuntimeError('Failed to write panorama: {}'.format(panorama_path))
        result = {
            'panorama_file': panorama_path.name,
            'panorama_width': int(panorama.shape[1]),
            'panorama_height': int(panorama.shape[0]),
            'stitcher_mode': mode,
            'transform_method': 'independent SIFT/RANSAC image-to-panorama homography',
            'transform_note': ('Approximation fitted to final panorama pixels; '
                               'not the internal Stitcher warp. May be inaccurate '
                               'for spherical/cylindrical projection or parallax.'),
            'images': {}
        }
        for path, image in zip(paths, images):
            H, matches, inliers = self.estimate_transform(image, panorama)
            result['images'][path.name] = {
                'homography': H.tolist() if H is not None else None,
                'status': 'ok' if H is not None else 'insufficient_matches',
                'good_matches': matches,
                'inliers': inliers,
                'source_width': int(image.shape[1]),
                'source_height': int(image.shape[0])
            }
            rospy.loginfo('%s: %s (%d inliers)', path.name,
                          result['images'][path.name]['status'], inliers)
        with json_path.open('w') as f:
            json.dump(result, f, indent=2)
        rospy.loginfo('Saved panorama: %s', panorama_path)
        rospy.loginfo('Saved transformations: %s', json_path)


if __name__ == '__main__':
    rospy.init_node('panorama_node')
    try:
        PanoramaNode()
    except Exception as exc:
        rospy.logerr('Panorama processing failed: %s', exc)
        raise
