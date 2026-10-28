#!/usr/bin/env python3

import cv2
import json
import numpy as np
import rospy
import subprocess
from pathlib import Path
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32


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


class qualityControl:

    NUM_IMAGES = 6

    def __init__(self):
        
        self.bridge = CvBridge()

        self.remote_user = 'anamarija'
        self.remote_host = '178.172.42.49'
        self.remote_tmp = '/home/anamarija/SuperSimpleNet/tools/tmpServe'
        self.remote_serve = '/home/anamarija/SuperSimpleNet/tools/serve'
        self.base_dir = Path('/home/pal/docker_anamarija/dataForClothDefectService')

        self.result_timeout = float(rospy.get_param('~result_timeout', 120.0))
        self.feather_px = float(rospy.get_param('~feather_px', 35.0))
        self.remote_host = rospy.get_param('~remote_host', self.remote_host)
        if self.feather_px <= 0:
            raise ValueError('~feather_px must be positive')
        self.folder_name = None
        self.sample_idx = None
        self.sample_dir = None
        self.upload_dir = None
        self.images = []

        self.saved_pub = rospy.Publisher('/quality_control/image_saved', Bool, queue_size=1)
        rospy.Subscriber('/quality_control/trigger', Int32, self.trigger_callback)

    def trigger_callback(self, msg):

        idx = msg.data

        try:
            if idx == 1:
                self.images = []
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
        cv2.imwrite(str(self.sample_dir / f'image_{self.sample_idx:04d}_{idx}.png'), cv_img)

    def make_panorama(self):
        """Reconstruct the 1066x720 panorama using fixed image-to-panorama homographies."""
        if len(self.images) != self.NUM_IMAGES:
            raise RuntimeError('Expected 6 images, got %d' % len(self.images))

        numerator = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
        denominator = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

        for idx, (image, matrix) in enumerate(zip(self.images, HOMOGRAPHIES), start=1):
            if image is None or image.shape[:2] != (480, 640):
                raise ValueError('Image %d must be 640x480' % idx)

            H = np.asarray(matrix, dtype=np.float64)
            warped = cv2.warpPerspective(
                image, H, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR
            )
            weights = feather_weights(480, 640, self.feather_px)
            warped_weights = cv2.warpPerspective(
                weights, H, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR
            )
            numerator += warped.astype(np.float32) * warped_weights[:, :, None]
            denominator += warped_weights

        panorama = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        valid = denominator > 1e-6
        panorama[valid] = np.clip(
            numerator[valid] / denominator[valid, None], 0, 255
        ).astype(np.uint8)
        rospy.loginfo('Panorama created from fixed homographies: %dx%d, coverage %.1f%%',
                      WIDTH, HEIGHT, 100.0 * np.mean(valid))
        return panorama

    def save_panorama(self, pano):
        
        output = self.upload_dir / f'image_{self.sample_idx:04d}.png'
        if not cv2.imwrite(str(output), pano):
            raise RuntimeError('Could not save panorama: %s' % output)
        rospy.loginfo(f'Saved panorama to {self.upload_dir}')

    def copy_folder_to_remote(self):

        upload_name = f'upload_{self.folder_name}'
        target = f'{self.remote_user}@{self.remote_host}'
        staging = f'{self.remote_tmp}/{upload_name}'

        if not self.remote_host:
            raise RuntimeError('Remote host is empty; set ~remote_host before uploading')
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
