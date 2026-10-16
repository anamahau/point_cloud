#!/usr/bin/env python3

import cv2
import json
import rospy
import subprocess
from pathlib import Path
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class qualityControl:

    def __init__(self):

        self.bridge = CvBridge()

        self.remote_user = 'anamarija'
        self.remote_host = '178.172.42.49'
        self.remote_tmp = '/home/anamarija/SuperSimpleNet/tools/tmpServe'
        self.remote_serve = '/home/anamarija/SuperSimpleNet/tools/serve'
        self.base_dir = Path('/home/pal/docker_anamarija/dataForClothDefectService')

        self.result_timeout = 120.0   # seconds
        self.folder_name = None

        rospy.Subscriber('/quality_control/trigger', Bool, self.trigger_callback)


    def trigger_callback(self, msg):

        if not msg.data:
            return
        rospy.loginfo('Trigger received, running quality control...')
        try:
            self.main()
        except Exception as e:
            rospy.logerr(f'Quality control failed: {e}')


    def run_cmd(self, cmd):

        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            raise RuntimeError(f'Command failed: {cmd}')


    def save(self):

        rospy.loginfo('waiting for rgb_img message...')
        rgb_msg = rospy.wait_for_message('/rgbd/rgb/image_raw', Image, timeout=10)
        cv_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

        self.base_dir.mkdir(parents=True, exist_ok=True)
        next_idx = len(list(self.base_dir.glob('sample_*')))
        self.folder_name = f'sample_{next_idx:04d}'

        obs_dir = self.base_dir / self.folder_name / 'observation_start'
        obs_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(obs_dir / f'image_{next_idx:04d}.jpg'), cv_img)
        rospy.loginfo(f'Saved {self.folder_name}')


    def copy_folder_to_remote(self):

        # Upload under a name the remote watcher ignores (it only looks at sample_*),
        # then rename on the remote side. The rename is atomic, so the watcher
        # never sees a half-copied folder.
        upload_name = f'upload_{self.folder_name}'
        target = f'{self.remote_user}@{self.remote_host}'

        self.run_cmd(f'scp -r {self.base_dir}/{self.folder_name} {target}:{self.remote_tmp}/{upload_name}')
        self.run_cmd(f"ssh {target} 'mv {self.remote_tmp}/{upload_name} {self.remote_tmp}/{self.folder_name}'")


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
            f'{self.base_dir}/{self.folder_name}/'
        )


    def read_result(self):

        result_file = self.base_dir / self.folder_name / 'defect_predicted' / 'result.json'
        with open(result_file, 'r') as f:
            result = json.load(f)
        rospy.loginfo(f'Result: {result}')
        return result


    def main(self):

        self.save()
        self.copy_folder_to_remote()
        self.wait_for_result()
        self.copy_result_back()
        self.read_result()


if __name__ == '__main__':

    rospy.init_node('qualityControl_node')

    QC = qualityControl()
    rospy.loginfo('qualityControl node ready, waiting for trigger on /quality_control/trigger')
    rospy.spin()