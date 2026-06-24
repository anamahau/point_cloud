#!/usr/bin/env python3

import os
import cv2
import json
import rospy
import subprocess
import numpy as np
from std_msgs.msg import Bool



class qualityControl:

    def __init__(self):

        self.bridge = CvBridge()

        rospy.Subscriber('/quality_control/trigger', Bool, self.trigger_callback)

        self.container_name = 'talos_clothes'
        self.remote_user = 'anamarija'
        self.remote_host = '178.172.42.49'
        self.remote_path = '/home/anamarija/SuperSimpleNet/tools/cloth_defect_folder_service/serve'
        self.folder_path = '/home/pal/docker_anamarija/dataForClothDefectService'


    def trigger_callback(self, msg):

        rospy.loginfo('Trigger received, running quality control...')
        if msg.data:
            self.main()

        
    def removeBackground(rgbImg, depthImg):
        mask = ((depthImg > 0.0) & (depthImg < 0.8))
        rgb_masked = rgbImg.copy()
        rgb_masked[~mask] = [255, 255, 255]
        return rgb_masked


    def save(self):

        rospy.loginfo('waiting for rgb_img message...')
        self.rgb_img = rospy.wait_for_message('/rgbd/rgb/image_raw', Image, timeout=10)
        rospy.loginfo('waiting for depth_img message...')
        self.depth_img = rospy.wait_for_message('/rgbd/depth/image_raw', Image, timeout=10)

        self.base_dir = Path('/home/pal/docker_anamarija/dataForClothDefectService')
        existing = sorted(self.base_dir.glob('sample_*'))
        next_idx = len(existing)
        subfolder = Path('observation_start')
        self.new_folder = self.base_dir / f'sample_{next_idx:04d}' / subfolder
        self.new_folder.mkdir(parents=True, exist_ok=True)

        self.folder_name = f'sample_{next_idx:04d}'

        depth = self.bridge.imgmsg_to_cv2(self.depth_img, desired_encoding='passthrough')
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        elif depth.dtype == np.float32:
            pass
        cv_img = self.bridge.imgmsg_to_cv2(self.rgb_img, desired_encoding='bgr8')
        # TODO: remove background
        clothRGB = self.removeBackground(cv_img, depth)
        save_path = self.new_folder / f'image_{next_idx:04d}.jpg'
        cv2.imwrite(str(save_path), clothRGB)


    def run_cmd(self, cmd):

        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            raise RuntimeError(f'Command failed: {cmd}')


    def copy_folder_from_container():

        cmd = f'scp -r {self.folder_path}/{self.folder_name}/ {self.remote_user}@{self.remote_host}:{self.remote_path}'
        run_cmd(cmd)


    def wait_for_result():

        remote_json = f'{self.remote_path}/{self.folder_name}/defect_predicted/result.json'
        while True:
            cmd = f"ssh {self.remote_user}@{self.remote_host} 'test -f {remote_json}'"
            result = subprocess.run(cmd, shell=True)
            if result.returncode == 0:
                rospy.loginfo('Result found!')
                break
            rospy.loginfo('Waiting for result...')
            rospy.sleep(1)


    def copy_result_back():

        cmd = f"scp -r {self.remote_user}@{self.remote_host}:{self.remote_path}/{self.folder_name}/defect_predicted/ {FOLDER_PATH}/{self.folder_name}/"
        run_cmd(cmd)


    def main(self):

        self.save()
        self.copy_folder_from_container()
        self.wait_for_result()
        self.copy_result_back()


if __name__ == '__main__':

    rospy.init_node('qualityControl_node')

    QC = qualityControl()