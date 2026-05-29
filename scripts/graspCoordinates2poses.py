#!/usr/bin/env python3

import cv2
import json
import rospy
import subprocess
import numpy as np
from PIL import Image
from tf_reader import getTfTransform


def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f'Command failed: {cmd}')
    
def copy_folder_to_container():
    cmd = f'scp -r {REMOTE_USER}@{REMOTE_HOST}:{REMOTE_PATH}/{FOLDER_NAME}/ {FOLDER_PATH}/'
    run_cmd(cmd)

def copy_result_back():
    cmd = f'scp -r {FOLDER_PATH}/{FOLDER_NAME}/grasp_predicted/test.png {REMOTE_USER}@{REMOTE_HOST}:{REMOTE_PATH}/{FOLDER_NAME}/grasp_predicted/'
    run_cmd(cmd)

def delete_folder_in_container():
    cmd = f'rm -r {FOLDER_PATH}/{FOLDER_NAME}/'
    run_cmd(cmd)

def read_goal_coordinates():
    copy_folder_to_container()
    depth_img = np.array(Image.open(f'{FOLDER_PATH}/{FOLDER_NAME}/observation_start/depth_map.tiff'))
    rgb_img = cv2.imread(f'{FOLDER_PATH}/{FOLDER_NAME}/observation_start/image_left.png')
    file = f'{FOLDER_PATH}/{FOLDER_NAME}/observation_start/camera_intrinsics.json'
    with open(file, 'r') as f:
        data = json.load(f)
    f = data["focal_lengths_in_pixels"]
    c = data["principal_point_in_pixels"]
    fx = f["fx"]
    fy = f["fy"]
    cx = c["cx"]
    cy = c["cy"]
    transformMatrix = getTfTransform('base_link', 'rgbd_depth_optical_frame')
    file = f'{FOLDER_PATH}/{FOLDER_NAME}/grasp_predicted/grasp_coordinates.json'
    with open(file, 'r') as f:
        data = json.load(f)
    print(FOLDER_NAME)
    for i, grasp in enumerate(data):
        u = int(grasp["u"])
        v = int(grasp["v"])
        z = depth_img[v, u]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        xyz = transformMatrix @ np.array([x, y, z, 1])
        if (z == 0.0):
            print('point', i, '-', xyz[:3], '- invalid depth')
            cv2.circle(rgb_img, (u, v), 5, (0, 0, 255), -1)
            cv2.putText(rgb_img, str(i), (u-5, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        elif (xyz[0] > 0.7):
            print('point', i, '-', xyz[:3], '- unreachable x')
            cv2.circle(rgb_img, (u, v), 5, (255, 0, 0), -1)
            cv2.putText(rgb_img, str(i), (u-5, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        else:
            print('point', i, '-', xyz[:3])
            cv2.circle(rgb_img, (u, v), 5, (0, 255, 0), -1)
            cv2.putText(rgb_img, str(i), (u-5, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(f'{FOLDER_PATH}/{FOLDER_NAME}/grasp_predicted/test.png', rgb_img)
    copy_result_back()
    delete_folder_in_container()


CONTAINER_NAME = 'talos_clothes'
REMOTE_USER = 'anamarija'
REMOTE_HOST = '178.172.42.49'
REMOTE_PATH = '/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference'
FOLDER_PATH = '/talos_ws/dataForCedirnet'
FOLDER_NAME = 'sample_000021_2'


if __name__ == '__main__':

    rospy.init_node('graspCoordinates2poses_node', anonymous=True)

    read_goal_coordinates()