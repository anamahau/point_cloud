#!/usr/bin/env python3

import os
import time
import rospy
import subprocess
from std_msgs.msg import String, Bool


REMOTE_USER = 'anamarija'
REMOTE_HOST = '178.172.42.49'
REMOTE_PATH = '/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference'
FOLDER_NAME = None # read from ros topic


def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    
def copy_folder_to_remote():
    cmd = f"scp -r {FOLDER_NAME} {REMOTE_USER}@{REMOTE_HOST}:{REMOTE_PATH}/"
    run_cmd(cmd)

def wait_for_result():
    remote_json = f"{REMOTE_PATH}/{FOLDER_NAME}/grasp_predicted/grasp_pose.json"
    while True:
        cmd = f"ssh {REMOTE_USER}@{REMOTE_HOST} 'test -f {remote_json}'"
        result = subprocess.run(cmd, shell=True)
        if result.returncode == 0:
            print('Result found!')
            break
        print('Waiting for result...')
        time.sleep(1)

def copy_result_back():
    cmd = f"scp -r {REMOTE_USER}@{REMOTE_HOST}:{REMOTE_PATH}/{FOLDER_NAME}/grasp_predicted/ {FOLDER_NAME}/"

def folder_callback(msg):
    global FOLDER_NAME
    FOLDER_NAME = msg.data
    rospy.loginfo(f"Folder set to: {FOLDER_NAME}")
    # print(FOLDER_NAME)

def trigger_callback(msg):
    global FOLDER_NAME
    if not msg.data:
        return
    if FOLDER_NAME is None:
        rospy.logwarn('Trigger received but FOLDER_NAME is not set!')
        return
    rospy.loginfo(f"Trigger received. Processing {FOLDER_NAME}")
    try:
        copy_folder_to_remote()
        wait_for_result()
        copy_result_back()
        rospy.loginfo('Done :)')
    except Exception as e:
        rospy.logerr(f"Error: {e}")


if __name__ == '__main__':

    print(':)')

    rospy.init_node('sshBridge_node', anonymous=True)

    rospy.Subscriber('/cedirnet/folder_name', String, folder_callback)
    rospy.Subscriber('/cedirnet/trigger', Bool, trigger_callback)

    rospy.loginfo('SSH bridge node ready.')
    rospy.spin()