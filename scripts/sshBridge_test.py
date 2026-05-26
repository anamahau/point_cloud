#!/usr/bin/env python3

import subprocess
import time
import os
import rospy
from std_msgs.msg import String, Bool
import json
from geometry_msgs.msg import PoseStamped
import tf.transformations as tft


CONTAINER_NAME = 'talos_clothes'
REMOTE_USER = 'anamarija'
REMOTE_HOST = '178.172.42.49'
REMOTE_PATH = '/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference'

FOLDER_NAME = None
# FOLDER_NAME = 'sample_000011'
FOLDER_PATH = '/talos_ws/dataForCedirnet'
# FOLDER_PATH = '/home/pal/docker_anamarija/dataForCedirnet'


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
        rospy.logwarn('Trigger received but FOLDER_NAME not set!')
        return

    rospy.loginfo(f'Processing {FOLDER_NAME}')

    try:
        copy_folder_from_container()
        # copy_folder_to_container()
        wait_for_result()
        copy_result_back()
        rospy.loginfo('Done.')

    except Exception as e:
        rospy.logerr(f'Error: {e}')


# =========================
# MAIN
# =========================

if __name__ == '__main__':

    rospy.init_node('dockerBridge_node', anonymous=True)

    rospy.Subscriber('/cedirnet/folder_name', String, folder_callback)
    rospy.Subscriber('/cedirnet/trigger', Bool, trigger_callback)

    rospy.loginfo('Docker bridge node ready.')

    # copy_folder_from_container()
    # wait_for_result()
    # copy_result_back()

    rospy.spin()