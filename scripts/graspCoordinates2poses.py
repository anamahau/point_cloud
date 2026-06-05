#!/usr/bin/env python3

import cv2
import json
import rospy
import subprocess
import numpy as np
from PIL import Image
from tf_reader import getTfTransform
from scipy.spatial.transform import Rotation as R


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
    file = f'{FOLDER_PATH}/{FOLDER_NAME}/observation_start/camera_pose_in_world.json'
    with open(file, 'r') as f:
        data = json.load(f)
    roll = data["rotation_euler_xyz_in_radians"]["roll"]
    pitch = data["rotation_euler_xyz_in_radians"]["pitch"]
    yaw = data["rotation_euler_xyz_in_radians"]["yaw"]
    rotation = R.from_euler('xyz', [roll, pitch, yaw])
    rotation_matrix = rotation.as_matrix()
    transformMatrix_base2orb = np.eye(4)
    transformMatrix_base2orb[:3, :3] = rotation_matrix
    transformMatrix_base2orb[:3, 3] = [
        data["position_in_meters"]["x"],
        data["position_in_meters"]["y"],
        data["position_in_meters"]["z"]
    ]
    transformMatrix_orb2rs = np.array([
            [ 0.9989575 , 0.04022338,  0.02158668, -0.00320142],
            [-0.03988856, 0.99908041, -0.0157236 , -0.10964579],
            [-0.02219929, 0.01484615,  0.99964333, -0.05616454],
            [ 0.0       , 0.0       ,  0.0       ,  1.0       ]
        ])
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
        xyz_tmp = transformMatrix_orb2rs @ np.array([x, y, z, 1])
        xyz = transformMatrix_base2orb @ xyz_tmp
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
FOLDER_NAME = 'sample_000027'


if __name__ == '__main__':

    rospy.init_node('graspCoordinates2poses_node', anonymous=True)

    read_goal_coordinates()