#!/usr/bin/env python3
"""ROS Noetic quality control with calibrated spherical multiband panorama.

Requires spherical_calibration.json and spherical_maps.npz created beforehand.
"""
import json
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32


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



class QualityControlSpherical:
    NUM_IMAGES = 6

    def __init__(self):
        self.bridge = CvBridge()
        self.base_dir = Path(rospy.get_param(
            '~base_dir', '/home/pal/docker_anamarija/dataForClothDefectService'))
        self.calibration_dir = Path(rospy.get_param(
            '~calibration_dir', '/talos_ws/samples/spherical_calibration'))
        self.feather_px = float(rospy.get_param('~feather_px', 250.0))
        self.blend_levels = int(rospy.get_param('~blend_levels', 10))
        self.capture_timeout = float(rospy.get_param('~capture_timeout', 10.0))
        self.capture_delay = float(rospy.get_param('~capture_delay', 0.0))
        self.result_timeout = float(rospy.get_param('~result_timeout', 120.0))
        self.remote_user = rospy.get_param('~remote_user', 'anamarija')
        self.remote_host = rospy.get_param('~remote_host', '178.172.42.49')
        self.remote_tmp = rospy.get_param(
            '~remote_tmp', '/home/anamarija/SuperSimpleNet/tools/tmpServe')
        self.remote_serve = rospy.get_param(
            '~remote_serve', '/home/anamarija/SuperSimpleNet/tools/serve')
        self.image_topic = rospy.get_param('~image_topic', '/rgbd/rgb/image_raw')
        self.sample_idx = None
        self.folder_name = None
        self.sample_dir = None
        self.upload_dir = None
        self.captured = set()
        self.processing = False
        self.saved_pub = rospy.Publisher(
            '/quality_control/image_saved', Bool, queue_size=1)
        self.trigger_sub = rospy.Subscriber(
            '/quality_control/trigger', Int32, self.trigger_callback, queue_size=1)
        rospy.loginfo('Spherical quality control ready; waiting for triggers 1..6')

    def start_sample(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        existing = []
        for p in self.base_dir.iterdir():
            if p.is_dir():
                match = re.fullmatch(r'sample_(\d+)', p.name)
                if match:
                    existing.append(int(match.group(1)))
        self.sample_idx = max(existing, default=0) + 1
        self.folder_name = 'sample_%04d' % self.sample_idx
        self.sample_dir = self.base_dir / self.folder_name
        self.upload_dir = self.sample_dir / self.folder_name
        self.upload_dir.mkdir(parents=True, exist_ok=False)
        self.captured.clear()
        rospy.loginfo('Started %s', self.folder_name)

    def capture_image(self, idx):
        if self.capture_delay > 0:
            rospy.sleep(self.capture_delay)
        msg = rospy.wait_for_message(
            self.image_topic, Image, timeout=self.capture_timeout)
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        path = self.sample_dir / ('image_%04d_%d.png' % (self.sample_idx, idx))
        if not cv2.imwrite(str(path), image):
            raise RuntimeError('Failed to save captured image: %s' % path)
        self.captured.add(idx)
        rospy.loginfo('Saved image %d/%d: %s', idx, self.NUM_IMAGES, path)

    def load_images(self):
        paths = [self.sample_dir / ('image_%04d_%d.png' % (self.sample_idx, i))
                 for i in range(1, self.NUM_IMAGES + 1)]
        images = [cv2.imread(str(p)) for p in paths]
        for p, image in zip(paths, images):
            if image is None:
                raise RuntimeError('Cannot read captured image: %s' % p)
        return images

    def make_panorama(self):
        return build_panorama(self.load_images(), self.calibration_dir,
                              self.feather_px, self.blend_levels)

    def save_panorama(self, panorama):
        path = self.upload_dir / ('image_%04d.png' % self.sample_idx)
        if not cv2.imwrite(str(path), panorama):
            raise RuntimeError('Failed to save panorama: %s' % path)
        rospy.loginfo('Saved spherical panorama: %s (%dx%d)',
                      path, panorama.shape[1], panorama.shape[0])

    def remote_target(self):
        if not self.remote_host:
            raise RuntimeError('Set ~remote_host before uploading')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', self.remote_user):
            raise ValueError('Invalid remote_user')
        if not re.fullmatch(r'[A-Za-z0-9_.:-]+', self.remote_host):
            raise ValueError('Invalid remote_host')
        return '%s@%s' % (self.remote_user, self.remote_host)

    @staticmethod
    def run_cmd(args, timeout=60):
        subprocess.run(args, check=True, timeout=timeout)

    def copy_folder_to_remote(self):
        target = self.remote_target()
        staging = self.remote_tmp.rstrip('/') + '/upload_' + self.folder_name
        destination = self.remote_tmp.rstrip('/') + '/' + self.folder_name
        # Keep the same observation_start/sample_XXXX structure as qualityControl_v03.
        self.run_cmd(['ssh', target, 'mkdir -p -- ' + staging])
        self.run_cmd(['scp', '-r', str(self.upload_dir),
                      target + ':' + staging + '/observation_start'])
        self.run_cmd(['ssh', target, 'mv -- ' + staging + ' ' + destination])
        rospy.loginfo('Uploaded %s to %s', self.folder_name, target)

    def wait_for_result(self):
        target = self.remote_target()
        remote_json = (self.remote_serve.rstrip('/') + '/' + self.folder_name +
                       '/defect_predicted/result.json')
        deadline = time.monotonic() + self.result_timeout
        while not rospy.is_shutdown():
            check = subprocess.run(['ssh', '-o', 'BatchMode=yes',
                                    '-o', 'ConnectTimeout=5', target,
                                    'test -f ' + remote_json],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=10)
            if check.returncode == 0:
                rospy.loginfo('Remote result found: %s', remote_json)
                return
            if time.monotonic() >= deadline:
                raise RuntimeError('Timed out waiting for %s' % remote_json)
            rospy.sleep(1.0)
        raise rospy.ROSInterruptException('Shutdown while waiting for result')

    def copy_result_back(self):
        remote_path = (self.remote_serve.rstrip('/') + '/' + self.folder_name +
                       '/defect_predicted')
        self.run_cmd(['scp', '-r', self.remote_target() + ':' + remote_path,
                      str(self.sample_dir) + '/'])
        rospy.loginfo('Downloaded prediction results to %s', self.sample_dir)

    def read_result(self):
        result_path = self.sample_dir / 'defect_predicted' / 'result.json'
        with result_path.open() as f:
            result = json.load(f)
        rospy.loginfo('Quality-control result: %s', json.dumps(result))
        return result

    def process(self):
        panorama = self.make_panorama()
        self.save_panorama(panorama)
        self.copy_folder_to_remote()
        self.wait_for_result()
        self.copy_result_back()
        return self.read_result()

    def trigger_callback(self, msg):
        idx = int(msg.data)
        if self.processing:
            rospy.logwarn('Ignoring trigger %d: processing in progress', idx)
            return
        if not 1 <= idx <= self.NUM_IMAGES:
            rospy.logwarn('Ignoring invalid trigger index %d', idx)
            return
        if idx == 1:
            if self.captured and len(self.captured) < self.NUM_IMAGES:
                rospy.logwarn('Restarting incomplete sample')
            try:
                self.start_sample()
            except Exception as exc:
                rospy.logerr('Cannot start sample: %s', exc)
                return
        if self.sample_dir is None:
            rospy.logwarn('Trigger %d ignored: trigger 1 required first', idx)
            return
        expected = len(self.captured) + 1
        if idx != expected:
            rospy.logwarn('Trigger %d ignored: expected %d', idx, expected)
            return
        try:
            self.capture_image(idx)
            self.saved_pub.publish(Bool(data=True))
            if len(self.captured) == self.NUM_IMAGES:
                self.processing = True
                try:
                    self.process()
                    rospy.loginfo('Quality control completed for %s', self.folder_name)
                finally:
                    self.sample_dir = None
                    self.captured.clear()
                    self.processing = False
        except Exception as exc:
            rospy.logerr('Quality control failed: %s', exc)
            self.saved_pub.publish(Bool(data=False))
            if self.processing:
                self.sample_dir = None
                self.captured.clear()
                self.processing = False


def main():
    rospy.init_node('qualityControl_spherical_node')
    QualityControlSpherical()
    rospy.spin()


if __name__ == '__main__':
    main()
