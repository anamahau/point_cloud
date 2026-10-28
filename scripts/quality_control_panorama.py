#!/usr/bin/env python3

import os
import re
from pathlib import Path

import rospy
import json
import cv2
import numpy as np


class PanoramaNode:

    def __init__(self):
        rospy.init_node('panorama_node')

        self.input_dir = Path(
            rospy.get_param('~input_dir', 'sample_0075')
        )

        self.num_images = rospy.get_param('~num_images', 6)
        self.ratio = rospy.get_param('~ratio_threshold', 0.75)
        self.ransac_threshold = rospy.get_param(
            '~ransac_threshold', 4.0
        )

        self.process()

    def load_images(self):
        paths = sorted(
            self.input_dir.glob('image_*.png'),
            key=lambda p: int(p.stem.split('_')[-1])
        )

        if len(paths) != self.num_images:
            raise ValueError(
                f'Expected {self.num_images} images, '
                f'found {len(paths)}'
            )

        images = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f'Cannot read {path}')
            images.append(image)

        return paths, images

    def estimate_homography(self, img1, img2):
        """Estimate transformation from img1 to img2."""

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create(nfeatures=5000)

        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            raise RuntimeError('No descriptors detected')

        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = matcher.knnMatch(des1, des2, k=2)

        good = [
            m for m, n in matches
            if m.distance < self.ratio * n.distance
        ]

        if len(good) < 4:
            raise RuntimeError(
                f'Not enough matches: {len(good)}'
            )

        src = np.float32([
            kp1[m.queryIdx].pt for m in good
        ]).reshape(-1, 1, 2)

        dst = np.float32([
            kp2[m.trainIdx].pt for m in good
        ]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src,
            dst,
            cv2.RANSAC,
            self.ransac_threshold
        )

        if H is None or np.count_nonzero(mask) < 4:
            raise RuntimeError('Homography estimation failed')

        return H

    def calculate_transforms(self, images):
        """
        Express all transformations in the
        coordinate system of the middle image.
        """

        n = len(images)
        reference = n // 2

        transforms = [None] * n
        transforms[reference] = np.eye(3)

        # Images before reference
        for i in range(reference - 1, -1, -1):
            H = self.estimate_homography(
                images[i], images[i + 1]
            )
            transforms[i] = transforms[i + 1] @ H

        # Images after reference
        for i in range(reference + 1, n):
            H = self.estimate_homography(
                images[i], images[i - 1]
            )
            transforms[i] = transforms[i - 1] @ H

        return transforms, reference

    def create_panorama(self, images, transforms):
        """Warp images and blend overlapping pixels."""

        all_corners = []

        for image, H in zip(images, transforms):
            h, w = image.shape[:2]

            corners = np.float32([
                [0, 0], [w, 0],
                [w, h], [0, h]
            ]).reshape(-1, 1, 2)

            warped = cv2.perspectiveTransform(
                corners, H
            )
            all_corners.append(warped)

        points = np.concatenate(all_corners, axis=0)

        xmin, ymin = np.floor(
            points.min(axis=0)[0]
        ).astype(int)

        xmax, ymax = np.ceil(
            points.max(axis=0)[0]
        ).astype(int)

        width = int(xmax - xmin)
        height = int(ymax - ymin)

        translation = np.array([
            [1, 0, -xmin],
            [0, 1, -ymin],
            [0, 0, 1]
        ], dtype=np.float64)

        accumulator = np.zeros(
            (height, width, 3), dtype=np.float32
        )
        weights = np.zeros(
            (height, width), dtype=np.float32
        )

        final_transforms = []

        for image, H in zip(images, transforms):
            H_final = translation @ H
            final_transforms.append(H_final)

            warped = cv2.warpPerspective(
                image, H_final, (width, height)
            )

            mask = np.ones(
                image.shape[:2], dtype=np.uint8
            ) * 255

            warped_mask = cv2.warpPerspective(
                mask, H_final, (width, height),
                flags=cv2.INTER_NEAREST
            )

            valid = warped_mask > 0

            accumulator[valid] += warped[valid]
            weights[valid] += 1.0

        weights[weights == 0] = 1.0

        panorama = (
            accumulator / weights[:, :, None]
        ).clip(0, 255).astype(np.uint8)

        return panorama, final_transforms

    def process(self):
        try:
            paths, images = self.load_images()

            rospy.loginfo(
                f'Loaded {len(images)} images'
            )

            transforms, reference = (
                self.calculate_transforms(images)
            )

            panorama, final_transforms = (
                self.create_panorama(images, transforms)
            )

            sample_id = self.input_dir.name.replace(
                'sample_', ''
            )

            panorama_path = (
                self.input_dir /
                f'panorama_{sample_id}.png'
            )

            json_path = (
                self.input_dir /
                f'transformations_{sample_id}.json'
            )

            cv2.imwrite(str(panorama_path), panorama)

            data = {
                'reference_image': paths[reference].name,
                'panorama_width': panorama.shape[1],
                'panorama_height': panorama.shape[0],
                'transformation_type': 'homography',
                'coordinate_system': 'panorama_pixels',
                'images': {}
            }

            for path, H in zip(paths, final_transforms):
                data['images'][path.name] = {
                    'homography': H.tolist()
                }

            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)

            rospy.loginfo(
                f'Panorama saved: {panorama_path}'
            )
            rospy.loginfo(
                f'Transformations saved: {json_path}'
            )

        except Exception as e:
            rospy.logerr(str(e))


if __name__ == '__main__':
    try:
        node = PanoramaNode()
    except rospy.ROSInterruptException:
        pass
