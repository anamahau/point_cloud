#!/usr/bin/env python3

import os
import cv2
import json
import time
import rospy
import shutil
import subprocess
import numpy as np
import tf.transformations as tft
from tf_reader import getTfTransform, init_tf
from std_msgs.msg import String, Bool, Int32
from geometry_msgs.msg import PoseStamped
from panorama_image import stitch_three
from scipy.spatial.transform import Rotation
from tf.transformations import quaternion_from_euler


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


def generate1imageSample():
    global FOLDER_NAME

    num = int(FOLDER_NAME[-6:]) - 2
    FOLDER_NAME = f'sample_{(num+1):06d}_01'

    shutil.copytree(f'{FOLDER_PATH}/sample_{(num+1):06d}', f'{FOLDER_PATH}/{FOLDER_NAME}')

    cameraJson = f'{FOLDER_PATH}/{FOLDER_NAME}/observation_start/camera_intrinsics.json'
    with open(cameraJson, 'r') as f:
        data = json.load(f)
    data["image_resolution"]["width"] = 1024
    data["image_resolution"]["height"] = 1024
    with open(cameraJson, 'w') as f:
        json.dump(data, f, indent=4)

    modelJson = f'{FOLDER_PATH}/{FOLDER_NAME}/observation_start/requested_model.json'
    with open(modelJson, 'r') as f:
        data = json.load(f)
    data["requested_model"] = "v2+seg+rand_bg+cropping_talos"
    with open(modelJson, 'w') as f:
        json.dump(data, f, indent=4)

    rgb = cv2.imread(f'{FOLDER_PATH}/{FOLDER_NAME}/observation_start/image_left.png')
    depth = cv2.imread(f'{FOLDER_PATH}/{FOLDER_NAME}/observation_start/depth_map.tiff', cv2.IMREAD_UNCHANGED)

    rgb_small = cv2.resize(rgb, (480, 360))
    rgb_padded = cv2.copyMakeBorder(
        rgb_small,
        top=100,
        bottom=564,
        left=272,
        right=272,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    depth_small = cv2.resize(depth, (480, 360))
    depth_padded = cv2.copyMakeBorder(
        depth_small,
        top=100,
        bottom=564,
        left=272,
        right=272,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    
    cv2.imwrite(
        f'{FOLDER_PATH}/{FOLDER_NAME}/observation_start/image_left.png',
        rgb_padded
    )
    cv2.imwrite(
        f'{FOLDER_PATH}/{FOLDER_NAME}/observation_start/depth_map.tiff',
        depth_padded
    )

    shutil.rmtree(f'{FOLDER_PATH}/{FOLDER_NAME}/grasp_predicted/')

    rospy.loginfo(f'Processing {FOLDER_NAME}')
    copy_folder_from_container()
    wait_for_result()
    copy_result_back()
    rospy.loginfo('Done')

    # newSampleResultJson = f'{FOLDER_PATH}/{FOLDER_NAME}/grasp_predicted/grasp_pose.json'
    # with open(newSampleResultJson, 'r') as f:
    #     newSampleJsonData = json.load(f)
    # best_grasp = max(newSampleJsonData, key=lambda x: x["score"])
    # print('best grasp:', best_grasp)
    # pos = best_grasp["position_in_meters"]
    # rot = best_grasp["rotation_euler_xyz_in_radians"]
    # msg = PoseStamped()
    # msg.pose.position.x = pos["x"]
    # msg.pose.position.y = pos["y"]
    # msg.pose.position.z = pos["z"]
    # quat = tft.quaternion_from_euler(rot["roll"], rot["pitch"], rot["yaw"])
    # msg.pose.orientation.x = quat[0]
    # msg.pose.orientation.y = quat[1]
    # msg.pose.orientation.z = quat[2]
    # msg.pose.orientation.w = quat[3]
    # pub = rospy.Publisher('/cedirnet/goal_pose', PoseStamped, queue_size=10, latch=True)
    # pub.publish(msg)


def generatePanoramaSample():
    global FOLDER_NAME

    images = []
    depths = []

    num = int(FOLDER_NAME[-6:]) - 2

    for i in range(3):
        image = cv2.imread(f'{FOLDER_PATH}/sample_{(num+i):06d}/observation_start/image_left.png')
        images.append(image)
        depth = cv2.imread(f'{FOLDER_PATH}/sample_{(num+i):06d}/observation_start/depth_map.tiff', cv2.IMREAD_UNCHANGED)
        depths.append(depth)

    img1 = images[0]
    img2 = images[1]
    img3 = images[2]
    depth1 = depths[0]
    depth2 = depths[1]
    depth3 = depths[2]

    mergedSampleName = f'sample_{(num):06d}_{(num+2):06d}'

    rgb_panorama, depth_panorama = stitch_three(img1, img2, img3, depth1, depth2, depth3)

    shutil.copytree(f'{FOLDER_PATH}/sample_{(num+1):06d}', f'{FOLDER_PATH}/{mergedSampleName}')

    cameraJson = f'{FOLDER_PATH}/{mergedSampleName}/observation_start/camera_intrinsics.json'
    with open(cameraJson, 'r') as f:
        data = json.load(f)
    data["image_resolution"]["width"] = 1024
    data["image_resolution"]["height"] = 1024
    with open(cameraJson, 'w') as f:
        json.dump(data, f, indent=4)

    modelJson = f'{FOLDER_PATH}/{mergedSampleName}/observation_start/requested_model.json'
    with open(modelJson, 'r') as f:
        data = json.load(f)
    data["requested_model"] = "v2+seg+rand_bg+cropping_talos"
    with open(modelJson, 'w') as f:
        json.dump(data, f, indent=4)

    rgb_small = cv2.resize(rgb_panorama, (768, 768))
    rgb_padded = cv2.copyMakeBorder(
        rgb_small,
        top=0,
        bottom=256,
        left=128,
        right=128,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    depth_small = cv2.resize(depth_panorama, (768, 768))
    depth_padded = cv2.copyMakeBorder(
        depth_small,
        top=0,
        bottom=256,
        left=128,
        right=128,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    cv2.imwrite(
        f'{FOLDER_PATH}/{mergedSampleName}/observation_start/image_left.png',
        rgb_padded
    )
    cv2.imwrite(
        f'{FOLDER_PATH}/{mergedSampleName}/observation_start/depth_map.tiff',
        depth_padded
    )

    shutil.rmtree(f'{FOLDER_PATH}/{mergedSampleName}/grasp_predicted/')

    FOLDER_NAME = mergedSampleName
    rospy.loginfo(f'Processing {FOLDER_NAME}')
    copy_folder_from_container()
    wait_for_result()
    copy_result_back()
    rospy.loginfo('Done')

    panoramaCoordinatesJson = f'{FOLDER_PATH}/{FOLDER_NAME}/grasp_predicted/grasp_coordinates.json'
    with open(panoramaCoordinatesJson, 'r') as f:
        coordinatesData = json.load(f)
    best_grasp = max(coordinatesData, key=lambda x: x["score"])
    u_final = best_grasp["u"]
    v_final = best_grasp["v"]
    rgb10 = cv2.imread(f'{FOLDER_PATH}/{FOLDER_NAME}/observation_start/image_left.png')
    depth10 = cv2.imread(f'{FOLDER_PATH}/{FOLDER_NAME}/observation_start/depth_map.tiff', cv2.IMREAD_UNCHANGED)
    xyz_odom = coordinates2odom(f'{FOLDER_PATH}/{FOLDER_NAME}/observation_start/camera_intrinsics.json', f'{FOLDER_PATH}/{FOLDER_NAME}/observation_start/camera_pose_in_world.json', u_final, v_final, rgb10, depth10)
    print('****************************************')
    print(f'x: {xyz_odom[0]:.4f}, y: {xyz_odom[1]:.4f}, z: {xyz_odom[2]:.4f}')
    print('****************************************')
    roll = best_grasp["rotation_euler_xyz_in_radians"]["roll"]
    pitch = best_grasp["rotation_euler_xyz_in_radians"]["pitch"]
    yaw = best_grasp["rotation_euler_xyz_in_radians"]["yaw"]
    qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
    pose_msg = PoseStamped()
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.header.frame_id = "odom"
    pose_msg.pose.position.x = xyz_odom[0]
    pose_msg.pose.position.y = xyz_odom[1]
    pose_msg.pose.position.z = xyz_odom[2]
    pose_msg.pose.orientation.x = qx
    pose_msg.pose.orientation.y = qy
    pose_msg.pose.orientation.z = qz
    pose_msg.pose.orientation.w = qw
    pub = rospy.Publisher('/cedirnet/goal_pose', PoseStamped, queue_size=10)
    pub.publish(pose_msg)
    # img2_shape = (480, 640)
    # intrinsics_json_path = f'{FOLDER_PATH}/{FOLDER_NAME}/observation_start/camera_intrinsics.json'
    # xyz_odom = panorama_pixel_to_world(u_final, v_final, img2_shape, depth2, intrinsics_json_path)
    # print('==== final point:', xyz_odom)

    # panoramaResultJson = f'{FOLDER_PATH}/{FOLDER_NAME}/grasp_predicted/grasp_pose.json'
    # with open(panoramaResultJson, 'r') as f:
    #     panoramaJsonData = json.load(f)
    # best_grasp = max(panoramaJsonData, key=lambda x: x["score"])
    # print('best grasp:', best_grasp)
    # pos = best_grasp["position_in_meters"]
    # rot = best_grasp["rotation_euler_xyz_in_radians"]
    # msg = PoseStamped()
    # msg.pose.position.x = pos["x"]
    # msg.pose.position.y = pos["y"]
    # msg.pose.position.z = pos["z"]
    # quat = tft.quaternion_from_euler(rot["roll"], rot["pitch"], rot["yaw"])
    # msg.pose.orientation.x = quat[0]
    # msg.pose.orientation.y = quat[1]
    # msg.pose.orientation.z = quat[2]
    # msg.pose.orientation.w = quat[3]
    # pub = rospy.Publisher('/cedirnet/goal_pose', PoseStamped, queue_size=10, latch=True)
    # pub.publish(msg)
    # print('message published')


'''def coordinates2poseForNewSample(sample, depth):
    path = f'{FOLDER_PATH}/{sample}/grasp_predicted/grasp_coordinates.json'
    with open(path, 'r') as f:
        coordinatesData = json.load(f)
    best_grasp = max(coordinatesData, key=lambda x: x["score"])
    print('best grasp:', best_grasp)
    path = f'{FOLDER_PATH}/{sample}/observation_start/camera_intrinsics.json'
    with open(path, 'r') as f:
        intrinsicsData = json.load(f)
    u = best_grasp["u"]
    v = best_grasp["v"]
    z = depth[v, u]
    u = u - 271
    v = v - 203
    path = f'{FOLDER_PATH}/{sample}/observation_start/camera_pose_in_world.json'
    transformMatrix_base2orb = load_transformation_matrix(path)
    transformMatrix_orb2rs = np.array([
            [ 0.9989575 , 0.04022338,  0.02158668, -0.00320142],
            [-0.03988856, 0.99908041, -0.0157236 , -0.10964579],
            [-0.02219929, 0.01484615,  0.99964333, -0.05616454],
            [ 0.0       , 0.0       ,  0.0       ,  1.0       ]
        ])
    fx = intrinsicsData["fx"]
    fy = intrinsicsData["fy"]
    cx = intrinsicsData["cx"]
    cy = intrinsicsData["cy"]

    if v < 360:
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        xyz_tmp = transformMatrix_orb2rs @ np.array([x, y, z, 1])
        xyz = transformMatrix_base2orb @ xyz_tmp
        print('best grasp point -', xyz)'''


def getOriginalFromPanorama(rgb, depth, u, v):

    rgb_crop = rgb[204:564, 272:752]
    u = int(u - 272)
    v = int(v - 204)
    rgb_crop_resized = cv2.resize(rgb_crop, (640, 480))
    u = int(u / 0.75)
    v = int(v / 0.75)

    depth_crop = depth[204:564, 272:752]
    depth_crop_resized = cv2.resize(depth_crop, (640, 480))

    return rgb_crop_resized, depth_crop_resized


def recalculateCoordinates(u, v):
    u = int((u - 272) / 0.75)
    v = int((v - 204) / 0.75)
    return u, v


def coordinates2odom(intrinsicsPath, cameraPosePath, u, v, rgb, depth):
    with open(intrinsicsPath, 'r') as f:
        cameraIntrinsics = json.load(f)
    fx = cameraIntrinsics["focal_lengths_in_pixels"]["fx"]
    fy = cameraIntrinsics["focal_lengths_in_pixels"]["fy"]
    cx = cameraIntrinsics["principal_point_in_pixels"]["cx"]
    cy = cameraIntrinsics["principal_point_in_pixels"]["cy"]

    rgb_original, depth_original = getOriginalFromPanorama(rgb, depth, u, v)
    u_new, v_new = recalculateCoordinates(u, v)
    print(f'new: {u_new}, {v_new}')
    v_new = min(v_new, 479)

    z = depth[v, u]
    z2 = depth_original[v_new, u_new]
    x = (u_new - cx) * z / fx
    y = (v_new - cy) * z / fy
    print(f'in rs CS -> x: {x}, y: {y}, z: {z} (z2: {z2})')

    transform_base2orb = getMatrixFromJson(cameraPosePath)
    transform_orb2rs = np.array([
        [ 0.9989575 , 0.04022338,  0.02158668, -0.00320142],
        [-0.03988856, 0.99908041, -0.0157236 , -0.10964579],
        [-0.02219929, 0.01484615,  0.99964333, -0.05616454],
        [ 0.0       , 0.0       ,  0.0       ,  1.0       ]
    ])
    xyz_cam = transform_orb2rs @ np.array([x, y, z, 1.0])
    print(f'in orb CS -> x: {xyz_cam[0]}, y: {xyz_cam[1]}, z: {xyz_cam[2]}')
    xyz_world = transform_base2orb @ xyz_cam
    print(f'in odom CS -> x: {xyz_world[0]}, y: {xyz_world[1]}, z: {xyz_world[2]}\n')

    return [xyz_world[0], xyz_world[1], xyz_cam[2]]


def getMatrixFromJson(poseJsonPath):
    with open(poseJsonPath, 'r') as f:
        data = json.load(f)
    
    t = np.array([
        data["position_in_meters"]["x"],
        data["position_in_meters"]["y"],
        data["position_in_meters"]["z"]
    ])

    # --- extract Euler angles ---
    roll  = data["rotation_euler_xyz_in_radians"]["roll"]
    pitch = data["rotation_euler_xyz_in_radians"]["pitch"]
    yaw   = data["rotation_euler_xyz_in_radians"]["yaw"]

    # --- convert to rotation matrix ---
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw])
    R_mat = rot.as_matrix()

    # --- build homogeneous transform ---
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t

    # print(f'transform matrix:\n{T}')

    return(T)


def panorama_to_middle_pixel(u_final, v_final, img2_shape):
    """
    Map a pixel picked on the *saved* panorama image back to the pixel
    coordinates of the original, un-warped middle image (img2).

    Returns (u2, v2) or None if the point falls outside the middle image
    region (i.e. it's actually on a side image / black padding, which
    would violate the "point is always in the middle image" assumption).
    """

    MIDDLE_OFFSET_X = 192   # where img2 is pasted into the raw 1024x1024 canvas
    MIDDLE_OFFSET_Y = 272
    FINAL_SCALE     = 0.75  # 1024 -> 768 resize applied after stitching
    FINAL_PAD_LEFT  = 128
    FINAL_PAD_TOP   = 0

    h2, w2 = img2_shape[:2]  # expected 480, 640 for your raw camera resolution

    # 1. undo the final pad, then the 0.75 resize -> back to raw 1024x1024 canvas coords
    u_canvas = (u_final - FINAL_PAD_LEFT) / FINAL_SCALE
    v_canvas = (v_final - FINAL_PAD_TOP) / FINAL_SCALE

    # 2. undo the paste offset of img2 into the canvas
    u2 = u_canvas - MIDDLE_OFFSET_X
    v2 = v_canvas - MIDDLE_OFFSET_Y

    if not (0 <= u2 < w2 and 0 <= v2 < h2):
        return None

    return u2, v2


def pixel_to_world(u2, v2, depth_img2, intrinsics_json_path):
    """
    Same math as read_goal_coordinates() in sshBridge_test.py, applied to a
    single pixel in the *original* middle image (matches the intrinsics json).
    """

    TRANSFORM_ORB2RS = np.array([
        [ 0.9989575 , 0.04022338,  0.02158668, -0.00320142],
        [-0.03988856, 0.99908041, -0.0157236 , -0.10964579],
        [-0.02219929, 0.01484615,  0.99964333, -0.05616454],
        [ 0.0       , 0.0       ,  0.0       ,  1.0       ]
    ])

    with open(intrinsics_json_path, 'r') as f:
        data = json.load(f)

    fx = data['focal_lengths_in_pixels']['fx']
    fy = data['focal_lengths_in_pixels']['fy']
    cx = data['principal_point_in_pixels']['cx']
    cy = data['principal_point_in_pixels']['cy']

    u_int, v_int = int(round(u2)), int(round(v2))
    z = depth_img2[v_int, u_int]

    print('z:', z)

    print(u_int, v_int)
    print(np.shape(depth_img2))

    depth_norm = cv2.normalize(depth_img2, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)
    cv2.circle(depth_norm, (u_int, v_int), 15, 255, -1)
    cv2.circle(depth_norm, (u_int, v_int), 10, 0, -1)
    cv2.circle(depth_norm, (u_int, v_int), 5, 255, -1)
    cv2.imwrite(f'{FOLDER_PATH}/{FOLDER_NAME}/grasp_predicted/depth.png', depth_norm)

    if z == 0:
        raise ValueError(f'Invalid (zero) depth at pixel ({u_int}, {v_int})')

    x = (u_int - cx) * z / fx
    y = (v_int - cy) * z / fy

    path = f'{FOLDER_PATH}/{FOLDER_NAME}/observation_start/camera_pose_in_world.json'
    transform_base2orb = load_transformation_matrix(path)
    xyz_cam = TRANSFORM_ORB2RS @ np.array([x, y, z, 1.0])
    xyz_world = transform_base2orb @ xyz_cam

    return xyz_world[:3]


def panorama_pixel_to_world(u_final, v_final, img2_shape, depth_img2, intrinsics_json_path):
    """
    Full pipeline: a pixel clicked on the generated panorama -> world (x, y, z).
    Raises if the pixel doesn't land inside the middle-image region.
    """
    mapped = panorama_to_middle_pixel(u_final, v_final, img2_shape)
    if mapped is None:
        raise ValueError(
            f'Pixel ({u_final}, {v_final}) is outside the middle image region '
            'of the panorama - the single-image intrinsics/extrinsics '
            'assumption does not hold for this point.'
        )
    u2, v2 = mapped
    return pixel_to_world(u2, v2, depth_img2, intrinsics_json_path)


def load_transformation_matrix(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    t = np.array([
        data["position_in_meters"]["x"],
        data["position_in_meters"]["y"],
        data["position_in_meters"]["z"],
    ])

    roll = data["rotation_euler_xyz_in_radians"]["roll"]
    pitch = data["rotation_euler_xyz_in_radians"]["pitch"]
    yaw = data["rotation_euler_xyz_in_radians"]["yaw"]

    R = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


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

    finished_pub = rospy.Publisher('/cedirnet/finished', Bool, queue_size=10)

    try:
        copy_folder_from_container()
        # copy_folder_to_container()
        wait_for_result()
        copy_result_back()
        rospy.loginfo('Done')
        # read_goal_coordinates()
        # fill_merged_json(msg.data)
        finished_pub.publish(True)
        if (msg.data == 3):
            rospy.loginfo('Starting result transformation...')
            generatePanoramaSample()
            # generate1imageSample()

    except Exception as e:
        rospy.logerr(f'Error: {e}')


# =========================
# MAIN
# =========================

if __name__ == '__main__':

    rospy.init_node('dockerBridge_node', anonymous=True)

    init_tf()

    rospy.Subscriber('/cedirnet/folder_name', String, folder_callback)
    rospy.Subscriber('/cedirnet/trigger2', Int32, trigger_callback)

    rospy.loginfo('Docker bridge node ready.')

    # copy_folder_from_container()
    # wait_for_result()
    # copy_result_back()

    rospy.spin()