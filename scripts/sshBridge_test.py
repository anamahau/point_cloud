#!/usr/bin/env python3

import os
import cv2
import json
import time
import rospy
import subprocess
import numpy as np
import tf.transformations as tft
from tf_reader import getTfTransform
from std_msgs.msg import String, Bool, Int32
from geometry_msgs.msg import PoseStamped


CONTAINER_NAME = 'talos_clothes'
REMOTE_USER = 'anamarija'
REMOTE_HOST = '178.172.42.49'
REMOTE_PATH = '/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference'

FOLDER_NAME = None
# FOLDER_NAME = 'sample_000011'
# FOLDER_PATH = '/talos_ws/dataForCedirnet'
FOLDER_PATH = '/home/pal/docker_anamarija/dataForCedirnet'
MERGED_NAME = None


def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f'Command failed: {cmd}')


def copy_folder_to_container():
    cmd = f'docker cp {FOLDER_NAME} {CONTAINER_NAME}:{FOLDER_PATH}/'
    run_cmd(cmd)


def copy_folder_from_container():
    cmd = f'scp -r {FOLDER_PATH}/{FOLDER_NAME}/ {REMOTE_USER}@{REMOTE_HOST}:{REMOTE_PATH}'
    run_cmd(cmd)


def wait_for_result():
    remote_json = f'{REMOTE_PATH}/{FOLDER_NAME}/grasp_predicted/grasp_pose.json'
    while True:
        cmd = f"ssh {REMOTE_USER}@{REMOTE_HOST} 'test -f {remote_json}'"
        result = subprocess.run(cmd, shell=True)
        if result.returncode == 0:
            rospy.loginfo('Result found!')
            break
        rospy.loginfo('Waiting for result...')
        rospy.sleep(1)


def copy_result_back():
    cmd = f"scp -r {REMOTE_USER}@{REMOTE_HOST}:{REMOTE_PATH}/{FOLDER_NAME}/grasp_predicted/ {FOLDER_PATH}/{FOLDER_NAME}/"
    run_cmd(cmd)


def read_goal_pose():
    file = f"{FOLDER_PATH}/{FOLDER_NAME}/grasp_predicted/grasp_pose.json"
    with open(file, 'r') as f:
        data = json.load(f)
    print(json.dumps(data, indent=4))
    best_grasp = max(data, key=lambda x: x["score"])
    pos = best_grasp["position_in_meters"]
    rot = best_grasp["rotation_euler_xyz_in_radians"]
    msg = PoseStamped()
    # msg.header.stamp = rospy.Time.now()
    # msg.header.frame_id = 'odom'
    msg.pose.position.x = pos["x"]
    msg.pose.position.y = pos["y"]
    msg.pose.position.z = pos["z"]
    quat = tft.quaternion_from_euler(rot["roll"], rot["pitch"], rot["yaw"])
    msg.pose.orientation.x = quat[0]
    msg.pose.orientation.y = quat[1]
    msg.pose.orientation.z = quat[2]
    msg.pose.orientation.w = quat[3]
    pub = rospy.Publisher('/cedirnet/goal_pose', PoseStamped, queue_size=10)
    pub.publish(msg)

def read_goal_coordinates():
    depth_img = cv2.imread(f"{FOLDER_PATH}/{FOLDER_NAME}/observation_start/depth_map.tiff")
    rgb_img = cv2.imread(f"{FOLDER_PATH}/{FOLDER_NAME}/observation_start/image_left.png")
    file = f"{FOLDER_PATH}/{FOLDER_NAME}/observation_start/camera_intrinsics.json"
    with open(file, 'r') as f:
        data = json.load(f)
    f = data["focal_lengths_in_pixels"]
    c = data["principal_point_in_pixels"]
    fx = f["fx"]
    fy = f["fy"]
    cx = c["cx"]
    cy = c["cy"]
    # transformMatrix_base2orb = getTfTransform('base_link', 'rgbd_depth_optical_frame')
    transformMatrix_base2orb = getTfTransform('odom', 'rgbd_depth_optical_frame')
    transformMatrix_orb2rs = np.array([
            [ 0.9989575 , 0.04022338,  0.02158668, -0.00320142],
            [-0.03988856, 0.99908041, -0.0157236 , -0.10964579],
            [-0.02219929, 0.01484615,  0.99964333, -0.05616454],
            [ 0.0       , 0.0       ,  0.0       ,  1.0       ]
        ])
    file = f"{FOLDER_PATH}/{FOLDER_NAME}/grasp_predicted/grasp_coordinates.json"
    with open(file, 'r') as f:
        data = json.load(f)
    for i, grasp in enumerate(data):
        u = grasp["u"]
        v = grasp["v"]
        z = depth_img[v, u]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        xyz_tmp = transformMatrix_orb2rs @ np.array([x, y, z, 1])
        xyz = transformMatrix_base2orb @ xyz_tmp
        if (z == 0.0):
            print('point', i, '-', xyz, '- invalid depth')
            cv2.circle(rgb_img, (u, v), 5, (0, 0, 255), -1)
            cv2.putText(rgb_img, str(i), (u-5, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        elif (xyz[0] > 0.7):
            print('point', i, '-', xyz, '- unreachable x')
            cv2.circle(rgb_img, (u, v), 5, (255, 0, 0), -1)
            cv2.putText(rgb_img, str(i), (u-5, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        else:
            print('point', i, '-', xyz)
            cv2.circle(rgb_img, (u, v), 5, (0, 255, 0), -1)
            cv2.putText(rgb_img, str(i), (u-5, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(f"{FOLDER_PATH}/{FOLDER_NAME}/grasp_predicted/test.png", rgb_img)
    # msg = PoseStamped()
    # msg.pose.position.x = xyz[0]
    # msg.pose.position.y = xyz[1]
    # msg.pose.position.z = xyz[2]
    # pub = rospy.Publisher('/cedirnet/goal_pose', PoseStamped, queue_size=10)
    # pub.publish(msg)
    finished_pub = rospy.Publisher('/cedirnet/finished', Bool, queue_size=10)
    msgBool = Bool()
    msgBool.data = True
    finished_pub.publish(msgBool)

def fill_merged_json(msgNumber):
    global MERGED_NAME
    print('Hello from fill_merged_json function')
    print(f'msgNumber is {msgNumber}')
    if (msgNumber == 1):
        # if FOLDER_NAME is None:
        #     rospy.logwarn('Trigger for new .json file received, but FOLDER_NAME not set!')
        #     return
        firstNumber = int(FOLDER_NAME[-6:])
        MERGED_NAME = f'{FOLDER_NAME}-{(firstNumber+2):06d}'
        rospy.loginfo(f'Creating new merged .json file {MERGED_NAME}.json')
        path = f'{FOLDER_PATH}/../mergedSamples/{MERGED_NAME}.json'
        cmd = f'touch {path}'
        run_cmd(cmd)
        with open(path, 'w') as f:
            json.dump([], f)

    if MERGED_NAME is None:
        raise RuntimeError('MERGED_NAME is None')
    mergedFile = f'{FOLDER_PATH}/../mergedSamples/{MERGED_NAME}.json'
    with open(mergedFile, 'r') as f:
        mergedJsonData = json.load(f)
    depth_img = cv2.imread(f'{FOLDER_PATH}/{FOLDER_NAME}/observation_start/depth_map.tiff', cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise RuntimeError(f'Failed to read {depth_path}')
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
    transformMatrix_base2orb = getTfTransform('odom', 'rgbd_depth_optical_frame')
    transformMatrix_orb2rs = np.array([
            [ 0.9989575 , 0.04022338,  0.02158668, -0.00320142],
            [-0.03988856, 0.99908041, -0.0157236 , -0.10964579],
            [-0.02219929, 0.01484615,  0.99964333, -0.05616454],
            [ 0.0       , 0.0       ,  0.0       ,  1.0       ]
        ])
    file = f"{FOLDER_PATH}/{FOLDER_NAME}/grasp_predicted/grasp_coordinates.json"
    with open(file, 'r') as f:
        data = json.load(f)

    for i, grasp in enumerate(data):
        u = grasp["u"]
        v = grasp["v"]
        z = depth_img[v, u]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        score = grasp["score"]
        orientation = grasp["rotation_euler_xyz_in_radians"]
        tmpDict = {
            "position_in_meters": {
                "x": float(x),
                "y": float(y),
                "z": float(z)
            },
            "rotation_euler_xyz_in_radians": {
                "roll": float(orientation["roll"]),
                "pitch": float(orientation["pitch"]),
                "yaw": float(orientation["yaw"])
            },
            "score": float(score),
            "sample_number": int(FOLDER_NAME[-6:])
        }
        xyz_tmp = transformMatrix_orb2rs @ np.array([x, y, z, 1])
        xyz = transformMatrix_base2orb @ xyz_tmp
        if (z == 0.0):
            print('point', i, '-', xyz, '- invalid depth')
            cv2.circle(rgb_img, (u, v), 5, (0, 0, 255), -1)
            cv2.putText(rgb_img, str(i), (u-5, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            tmpDict.update({
                "comment": 'invalid depth'
            })
        elif (xyz[0] > 0.7):
            print('point', i, '-', xyz, '- unreachable x')
            cv2.circle(rgb_img, (u, v), 5, (255, 0, 0), -1)
            cv2.putText(rgb_img, str(i), (u-5, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            tmpDict.update({
                "comment": 'unreachable x'
            })
        else:
            print('point', i, '-', xyz)
            cv2.circle(rgb_img, (u, v), 5, (0, 255, 0), -1)
            cv2.putText(rgb_img, str(i), (u-5, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            tmpDict.update({
                "comment": ''
            })
        mergedJsonData.append(tmpDict)
    with open(mergedFile, 'w') as f:
        json.dump(mergedJsonData, f, indent=4)
    cv2.imwrite(f"{FOLDER_PATH}/{FOLDER_NAME}/grasp_predicted/test.png", rgb_img)
    finished_pub = rospy.Publisher('/cedirnet/finished', Bool, queue_size=10)
    # msgBool = Bool()
    # msgBool.data = True
    finished_pub.publish(True)
    if (msgNumber == 3):
        read_merged_json()

def read_merged_json():
    mergedFile = f'{FOLDER_PATH}/../mergedSamples/{MERGED_NAME}.json'
    with open(mergedFile, 'r') as f:
        mergedJsonData = json.load(f)
    best_grasp = max(mergedJsonData, key=lambda x: x["score"])
    pos = best_grasp["position_in_meters"]
    rot = best_grasp["rotation_euler_xyz_in_radians"]
    msg = PoseStamped()
    # msg.header.stamp = rospy.Time.now()
    # msg.header.frame_id = 'odom'
    msg.pose.position.x = pos["x"]
    msg.pose.position.y = pos["y"]
    msg.pose.position.z = pos["z"]
    quat = tft.quaternion_from_euler(rot["roll"], rot["pitch"], rot["yaw"])
    msg.pose.orientation.x = quat[0]
    msg.pose.orientation.y = quat[1]
    msg.pose.orientation.z = quat[2]
    msg.pose.orientation.w = quat[3]
    pub = rospy.Publisher('/cedirnet/goal_pose', PoseStamped, queue_size=10)
    pub.publish(msg)


# =========================
# ROS CALLBACKS
# =========================

def folder_callback(msg):
    global FOLDER_NAME
    FOLDER_NAME = msg.data
    rospy.loginfo(f'Folder set to: {FOLDER_NAME}')


def trigger_callback(msg):
    global FOLDER_NAME

    if not msg.data:
        return

    if FOLDER_NAME is None:
        rospy.logwarn('Trigger received, but FOLDER_NAME not set!')
        return

    rospy.loginfo(f'Processing {FOLDER_NAME}')

    try:
        copy_folder_from_container()
        # copy_folder_to_container()
        wait_for_result()
        copy_result_back()
        rospy.loginfo('Done.')
        rospy.loginfo('Starting result transformation...')
        # read_goal_coordinates()
        fill_merged_json(msg.data)

    except Exception as e:
        rospy.logerr(f'Error: {e}')


# def newJson_callback(msg):
#     global FOLDER_NAME
#     global MERGED_NAME

#     if not msg.data:
#         return
    
#     if FOLDER_NAME is None:
#         rospy.logwarn('Trigger for new .json file received, but FOLDER_NAME not set!')
#         return
    
#     firstNumber = int(FOLDER_NAME[-6:])
#     MERGED_NAME = f'{FOLDER_NAME}-{(firstNumber+2):06d}'
    
#     rospy.loginfo(f'Creating new merged .json file {MERGED_NAME}.json')

#     path = f'{FOLDER_PATH}/mergedSamples/{MERGED_NAME}.json'
#     cmd = f'touch {path}'
#     run_cmd(cmd)
#     with open(path, 'w') as f:
#         json.dump({}, f)


# =========================
# MAIN
# =========================

if __name__ == '__main__':

    rospy.init_node('dockerBridge_node', anonymous=True)

    rospy.Subscriber('/cedirnet/folder_name', String, folder_callback)
    rospy.Subscriber('/cedirnet/trigger', Int32, trigger_callback)
    # rospy.Subscriber('/cedirnet/new_json', Bool, newJson_callback)

    rospy.loginfo('Docker bridge node ready.')

    # copy_folder_from_container()
    # wait_for_result()
    # copy_result_back()

    rospy.spin()