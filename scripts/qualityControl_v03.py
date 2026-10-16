#!/usr/bin/env python3

import cv2
import json
import rospy
import subprocess
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

        for mode in (cv2.Stitcher_PANORAMA, cv2.Stitcher_SCANS):
            stitcher = cv2.Stitcher_create(mode)
            status, pano = stitcher.stitch(self.images)
            if status == cv2.Stitcher_OK:
                rospy.loginfo(f'Panorama created (mode {mode}), size {pano.shape[1]}x{pano.shape[0]}')
                return pano
            rospy.logwarn(f'Stitching with mode {mode} failed, status {status}')
        raise RuntimeError('Panorama stitching failed')
    
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