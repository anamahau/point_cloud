#!/usr/bin/env python3

import cv2
import json
import rospy
import subprocess
import numpy as np
from pathlib import Path
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32


class qualityControl:

    NUM_IMAGES = 6

    def __init__(self):
        
        self.bridge = CvBridge()

        self.remote_user = 'anamarija'
        self.remote_host = '178.172.42.49'
        self.remote_tmp = '/home/anamarija/SuperSimpleNet/tools/tmpServe'
        self.remote_serve = '/home/anamarija/SuperSimpleNet/tools/serve'
        self.base_dir = Path('/home/pal/docker_anamarija/dataForClothDefectService')

        self.result_timeout = 120.0
        self.folder_name = None
        self.sample_idx = None
        self.sample_dir = None
        self.upload_dir = None
        self.images = []

        self.image_trigger_indices = []

        self.saved_pub = rospy.Publisher('/quality_control/image_saved', Bool, queue_size=1)
        rospy.Subscriber('/quality_control/trigger', Int32, self.trigger_callback)

    def trigger_callback(self, msg):

        idx = msg.data

        try:
            if idx == 1:
                self.images = []
                self.image_trigger_indices = []
                self.start_sample()
            
            if self.folder_name is None:
                rospy.logwarn(f'Trigger {idx} received but no sample started (missing trigger 1)')
                return
            
            self.capture_image(idx)

            self.saved_pub.publish(True)

            if len(self.images) == self.NUM_IMAGES:
                rospy.loginfo('All images collected, running quality control...')
                self.process()
            
        except Exception as e:
            rospy.logerr(f'Quality control failed: {e}')

    def run_cmd(self, cmd):

        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            raise RuntimeError(f'Command failed: {cmd}')
        
    def start_sample(self):

        self.base_dir.mkdir(parents=True, exist_ok=True)

        existing = [int(p.name.split('_')[1])
                    for p in self.base_dir.glob('sample_[0-9]*') if p.is_dir()]
        self.sample_idx = max(existing, default=0) + 1

        self.folder_name = f'sample_{self.sample_idx:04d}'
        self.sample_dir = self.base_dir / self.folder_name
        self.upload_dir = self.sample_dir / self.folder_name
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        rospy.loginfo(f'Started {self.folder_name}')

    def capture_image(self, idx):

        rospy.loginfo(f'Capturing image {idx}/{self.NUM_IMAGES}...')
        rgb_msg = rospy.wait_for_message('/rgbd/rgb/image_raw', Image, timeout=10)
        cv_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        self.images.append(cv_img)
        self.image_trigger_indices.append(idx)
        cv2.imwrite(str(self.sample_dir / f'image_{self.sample_idx:04d}_{idx}.png'), cv_img)

    def make_panorama_old(self):

        for mode in (cv2.Stitcher_PANORAMA, cv2.Stitcher_SCANS):
            stitcher = cv2.Stitcher_create(mode)
            status, pano = stitcher.stitch(self.images)
            if status == cv2.Stitcher_OK:
                rospy.loginfo(f'Panorama created (mode {mode}), size {pano.shape[1]}x{pano.shape[0]}')
                try:
                    self.save_transformations(stitcher, mode, pano)
                except Exception as e:
                    rospy.logwarn(f'Could not save image transformations: {e}')
                return pano
            rospy.logwarn(f'Stitching with mode {mode} failed, status {status}')
        raise RuntimeError('Panorama stitching failed')

    def make_panorama(self):
        """Estimate, save, and apply per-image camera transformations."""

        if len(self.images) < 2:
            raise RuntimeError("At least two images are required")

        data = self.estimate_transformations()
        self.save_transformations(data)

        return self.render_panorama(data)
    

    def estimate_transformations(self):
        """Estimate camera parameters using OpenCV's detailed pipeline."""

        images = self.images
        n = len(images)

        h, w = images[0].shape[:2]

        for i, img in enumerate(images):
            if img is None or img.shape[:2] != (h, w):
                raise RuntimeError(
                    f"Image {i} is empty or has inconsistent dimensions"
                )

        # Resize images for feature detection and camera estimation.
        work_megapix = 0.6
        work_scale = min(
            1.0,
            np.sqrt(work_megapix * 1e6 / (h * w))
        )

        finder = cv2.SIFT_create()
        features = []

        for img in images:
            small = cv2.resize(
                img,
                None,
                fx=work_scale,
                fy=work_scale,
                interpolation=cv2.INTER_LINEAR
            )

            features.append(
                cv2.detail.computeImageFeatures2(finder, small)
            )

        # Match image features.
        matcher = cv2.detail.BestOf2NearestMatcher_create(
            False, 0.3
        )

        try:
            pairwise_matches = matcher.apply2(features)
        finally:
            matcher.collectGarbage()

        # Keep the largest connected group of matching images.
        # OpenCV also filters features and pairwise matches in place.
        indices = cv2.detail.leaveBiggestComponent(
            features, pairwise_matches, 1.0
        )
        indices = np.asarray(indices).reshape(-1).astype(int).tolist()

        if len(indices) != n:
            raise RuntimeError(
                f"Only {len(indices)} of {n} images are connected "
                f"by feature matches. Selected image indices: {indices}"
            )

        # Estimate camera intrinsics and rotations.
        estimator = cv2.detail_HomographyBasedEstimator()

        ok, cameras = estimator.apply(
            features, pairwise_matches, None
        )

        if not ok or len(cameras) != n:
            raise RuntimeError("Camera parameter estimation failed")

        for cam in cameras:
            cam.R = np.asarray(cam.R, dtype=np.float32)

        # Refine camera parameters with bundle adjustment.
        adjuster = cv2.detail_BundleAdjusterRay()
        adjuster.setConfThresh(1.0)

        refine_mask = np.zeros((3, 3), dtype=np.uint8)
        refine_mask[0, 0] = 1
        refine_mask[0, 1] = 1
        refine_mask[0, 2] = 1
        refine_mask[1, 1] = 1
        refine_mask[1, 2] = 1

        adjuster.setRefinementMask(refine_mask)

        ok, cameras = adjuster.apply(
            features, pairwise_matches, cameras
        )

        if not ok:
            raise RuntimeError("Camera bundle adjustment failed")

        # Correct panorama orientation.
        rotations = [
            np.asarray(cam.R, dtype=np.float32).copy()
            for cam in cameras
        ]

        rotations = cv2.detail.waveCorrect(
            rotations, cv2.detail.WAVE_CORRECT_HORIZ
        )

        for cam, rotation in zip(cameras, rotations):
            cam.R = np.asarray(rotation, dtype=np.float32)

        # Use the median estimated focal length as the warper scale.
        focals = sorted(float(cam.focal) for cam in cameras)
        middle = len(focals) // 2

        if n % 2:
            scale_work = focals[middle]
        else:
            scale_work = (
                focals[middle - 1] + focals[middle]
            ) / 2.0

        if not np.isfinite(scale_work) or scale_work <= 0:
            raise RuntimeError("Invalid estimated focal length")

        # Convert parameters to full-resolution image coordinates,
        # because rendering will use the original full-resolution images.
        camera_data = []

        for position, (original_index, cam) in enumerate(
            zip(indices, cameras)
        ):
            camera_data.append({
                "image_index": int(original_index),
                "trigger_index": int(
                    self.image_trigger_indices[original_index]
                ),
                "focal": float(cam.focal / work_scale),
                "aspect": float(cam.aspect),
                "ppx": float(cam.ppx / work_scale),
                "ppy": float(cam.ppy / work_scale),
                "R": np.asarray(cam.R, dtype=float).tolist()
            })

        return {
            "folder": self.folder_name,
            "num_input_images": n,
            "image_size_wh": [int(w), int(h)],
            "work_scale": float(work_scale),
            "warped_image_scale": float(scale_work / work_scale),
            "warp_type": "spherical",
            "wave_correction": "horizontal",
            "blend_type": "multiband",
            "blend_bands": 5,
            "used_image_indices": indices,
            "cameras": camera_data
        }

    def save_transformations(self, data):
        """Save estimated per-image camera parameters to JSON."""

        path = (
            self.sample_dir /
            f"transformations_{self.sample_idx:04d}.json"
        )

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        rospy.loginfo(
            f"Saved camera transformations for "
            f"{len(data['cameras'])} images to {path}"
        )


    def render_panorama(self, data):
        """Generate a panorama using saved camera parameters."""

        if len(self.images) != data["num_input_images"]:
            raise RuntimeError(
                "Input image count does not match transformation file"
            )

        expected_size = tuple(data["image_size_wh"])

        for i, img in enumerate(self.images):
            if (img.shape[1], img.shape[0]) != expected_size:
                raise RuntimeError(
                    f"Image {i} dimensions do not match the saved parameters"
                )

        cameras = sorted(
            data["cameras"],
            key=lambda cam: cam["image_index"]
        )

        indices = [cam["image_index"] for cam in cameras]

        if indices != data["used_image_indices"]:
            raise RuntimeError(
                "Camera indices do not match the saved image indices"
            )

        scale = float(data["warped_image_scale"])
        warper = cv2.PyRotationWarper(data["warp_type"], scale)

        corners = []
        sizes = []
        warped_images = []
        warped_masks = []

        for cam in cameras:
            img = self.images[cam["image_index"]]

            focal = cam["focal"]
            aspect = cam["aspect"]
            ppx = cam["ppx"]
            ppy = cam["ppy"]

            K = np.array([
                [focal, 0.0, ppx],
                [0.0, focal * aspect, ppy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)

            R = np.asarray(cam["R"], dtype=np.float32)

            # corner, warped = warper.warp(
            #     img,
            #     K,
            #     R,
            #     cv2.INTER_LINEAR,
            #     cv2.BORDER_REFLECT
            # )
            if (
                not np.all(np.isfinite(K))
                or not np.all(np.isfinite(R))
                or focal <= 0
                or aspect <= 0
            ):
                raise RuntimeError(
                    f"Invalid camera parameters for image "
                    f"{cam['image_index']}: "
                    f"focal={focal}, aspect={aspect}, K={K}, R={R}"
                )

            rospy.loginfo(
                f"Warping image {cam['image_index']}: "
                f"focal={focal:.2f}, "
                f"scale={scale:.2f}"
            )
            _, warped_mask = warper.warp(
                mask,
                K,
                R,
                cv2.INTER_NEAREST,
                cv2.BORDER_CONSTANT
            )

            mask = np.full(img.shape[:2], 255, dtype=np.uint8)

            _, warped_mask = warper.warp(
                mask,
                K,
                R,
                cv2.INTER_NEAREST,
                cv2.BORDER_CONSTANT
            )

            corners.append(corner)
            sizes.append((warped.shape[1], warped.shape[0]))
            warped_images.append(warped)
            warped_masks.append(warped_mask)

        roi = cv2.detail.resultRoi(corners, sizes)

        blender = cv2.detail_MultiBandBlender()
        blender.setNumBands(int(data["blend_bands"]))
        blender.prepare(roi)

        for img, mask, corner in zip(
            warped_images, warped_masks, corners
        ):
            blender.feed(
                img.astype(np.int16),
                mask,
                corner
            )

        panorama, _ = blender.blend(None, None)
        panorama = np.clip(panorama, 0, 255).astype(np.uint8)

        rospy.loginfo(
            f"Panorama generated from saved transformations: "
            f"{panorama.shape[1]}x{panorama.shape[0]}"
        )

        return panorama


    def make_panorama_fixed(self, transformation_path=None):
        """Recreate a panorama without estimating camera parameters again."""

        if transformation_path is None:
            transformation_path = (
                self.sample_dir /
                f"transformations_{self.sample_idx:04d}.json"
            )

        with open(transformation_path, "r") as f:
            data = json.load(f)

        rospy.loginfo(
            f"Loading saved transformations from {transformation_path}"
        )

        return self.render_panorama(data)
    
    def save_transformations_old(self, stitcher, mode, pano):

        # Camera parameters estimated by the stitcher (rotation, focal length, principal point, ...,
        # after bundle adjustment and wave correction). They are given at the stitcher's work scale
        # and are listed in the order of 'used_image_indices' (0-based, = trigger number - 1).
        # cameras = stitcher.cameras()
        # component = [int(i) for i in stitcher.component()]

        # focals = sorted(float(cam.focal) for cam in cameras)
        # m = len(focals)
        # warped_image_scale = focals[m // 2] if m % 2 else 0.5 * (focals[m // 2 - 1] + focals[m // 2])

        h, w = self.images[0].shape[:2]
        # data = {
        #     'folder': self.folder_name,
        #     'stitcher_mode': 'PANORAMA' if mode == cv2.Stitcher_PANORAMA else 'SCANS',
        #     'stitcher_mode_value': int(mode),
        #     'num_input_images': len(self.images),
        #     'image_size_wh': [w, h],
        #     'used_image_indices': component,
        #     'work_scale': float(stitcher.workScale()),
        #     'registration_resol_mpx': float(stitcher.registrationResol()),
        #     'warped_image_scale': warped_image_scale,
        #     'cameras': [
        #         {
        #             'image_index': component[k],
        #             'focal': float(cam.focal),
        #             'aspect': float(cam.aspect),
        #             'ppx': float(cam.ppx),
        #             'ppy': float(cam.ppy),
        #             'R': [[float(v) for v in row] for row in cam.R],
        #             't': [float(v) for v in cam.t.reshape(-1)],
        #         }
        #         for k, cam in enumerate(cameras)
        #     ],
        # }
        data = {
            'folder': self.folder_name,
            'stitcher_mode': (
                'PANORAMA'
                if mode == cv2.Stitcher_PANORAMA
                else 'SCANS'
            ),
            'stitcher_mode_value': int(mode),
            'num_input_images': len(self.images),
            'image_size_wh': [w, h],
            'panorama_size_wh': [
                int(pano.shape[1]),
                int(pano.shape[0])
            ],
            'used_image_indices': list(range(len(self.images))),
            'camera_parameters_available': False
        }

        path = self.sample_dir / f'transformations_{self.sample_idx:04d}.json'
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        rospy.loginfo(f'Saved image transformations to {path}')

    def save_panorama(self, pano):
        
        cv2.imwrite(str(self.upload_dir / f'image_{self.sample_idx:04d}.png'), pano)
        rospy.loginfo(f'Saved panorama to {self.upload_dir}')

    def copy_folder_to_remote(self):

        upload_name = f'upload_{self.folder_name}'
        target = f'{self.remote_user}@{self.remote_host}'
        staging = f'{self.remote_tmp}/{upload_name}'

        self.run_cmd(f"ssh {target} 'mkdir -p {staging}'")
        self.run_cmd(f'scp -r {self.upload_dir} {target}:{staging}/observation_start')
        self.run_cmd(f"ssh {target} 'mv {staging} {self.remote_tmp}/{self.folder_name}'")
        # self.run_cmd(f'scp -r {self.upload_dir} {target}:{self.remote_tmp}/{self.folder_name}')

    def wait_for_result(self):

        remote_json = f'{self.remote_serve}/{self.folder_name}/defect_predicted/result.json'
        target = f'{self.remote_user}@{self.remote_host}'
        start = rospy.Time.now()

        while not rospy.is_shutdown():
            result = subprocess.run(f"ssh {target} 'test -f {remote_json}'", shell=True)
            if result.returncode == 0:
                rospy.loginfo('Result found!')
                return
            if (rospy.Time.now() - start).to_sec() > self.result_timeout:
                raise RuntimeError(f'Timeout waiting for {remote_json}')
            rospy.loginfo('Waiting for result...')
            rospy.sleep(1)

    def copy_result_back(self):

        target = f'{self.remote_user}@{self.remote_host}'
        self.run_cmd(
            f'scp -r {target}:{self.remote_serve}/{self.folder_name}/defect_predicted/ '
            f'{self.sample_dir}/'
        )

    def read_result(self):

        result_file = self.sample_dir / 'defect_predicted' / 'result.json'
        with open(result_file, 'r') as f:
            result = json.load(f)
        rospy.loginfo(f'Result: {result}')
        return result
    
    def process(self):

        pano = self.make_panorama()
        self.save_panorama(pano)
        self.copy_folder_to_remote()
        self.wait_for_result()
        self.copy_result_back()
        self.read_result()

if __name__ == '__main__':
    
    rospy.init_node('qualityControl_node')

    QC = qualityControl()
    rospy.loginfo('qualityControl node ready, waiting for trigger on /quality_control/trigger')
    rospy.spin()