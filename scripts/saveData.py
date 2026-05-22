#!/usr/bin/env python3

import cv2
import glob
import json
import rospy

import numpy as np
import open3d as o3d
import tf.transformations as tft
import sensor_msgs.point_cloud2 as pc2

from pathlib import Path
from cv_bridge import CvBridge
from tf_reader import getTfTransform
from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState, Image, PointCloud2, CameraInfo


class dataRecorder:

    def __init__(self):
        
        rospy.init_node('data_recorder')

        self.folder_name_pub = rospy.Publisher('/cedirnet/folder_name', String, queue_size=10)
        self.cedirnet_trigger_pub = rospy.Publisher('/cedirnet/trigger', Bool, queue_size=10)

        self.bridge = CvBridge()

        rospy.Subscriber('/data_recorder/trigger', Bool, self.trigger_callback)


    def trigger_callback(self, msg):
        if msg.data:
            rospy.loginfo('Trigger received, running data recording...')
            self.save()
    

    def save(self):

        rospy.loginfo('waiting for joint_states message...')
        self.joint_states = rospy.wait_for_message('/joint_states', JointState, timeout=10)
        rospy.loginfo('waiting for depth_img message...')
        self.depth_img = rospy.wait_for_message('/rgbd/depth/image_raw', Image, timeout=10)
        rospy.loginfo('waiting for rgb_img message...')
        self.rgb_img = rospy.wait_for_message('/rgbd/rgb/image_raw', Image, timeout=10)
        rospy.loginfo('waiting for samera_info message...')
        self.camera_info = rospy.wait_for_message('/rgbd/rgb/camera_info', CameraInfo, timeout=10)
        rospy.loginfo('waiting for points message...')
        self.points_msg = rospy.wait_for_message('/rgbd/depth/points', PointCloud2, timeout=10)

        self.base_dir = Path('/talos_ws/dataForCedirnet')
        existing = sorted(self.base_dir.glob('sample_*'))
        next_idx = len(existing)
        subfolder = Path('observation_start')
        self.new_folder = self.base_dir / f'sample_{next_idx:06d}' / subfolder
        self.new_folder.mkdir(parents=True, exist_ok=True)

        self.folder_name = f'sample_{next_idx:06d}'
        self.folder_name_pub.publish(self.folder_name)

        # --------------------
        # arm_left_joints.json
        # --------------------
        left_joints = self.joint_states.position[0:7]
        data = {
            "values": list(left_joints)
        }
        save_path = self.new_folder / 'arm_left_joints.json'
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)

        # --------------------
        # arm_left_pose_in_world.json
        # --------------------
        tf = getTfTransform('base_link', 'arm_left_1_link', returnMatrix=False)
        tf_T = tf[0]
        tf_Q = tf[1]
        roll, pitch, yaw = tft.euler_from_quaternion(tf_Q)
        data = {
            "rotation_euler_xyz_in_radians": {
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw
            },
            "position_in_meters": {
                "x": tf_T[0],
                "y": tf_T[1],
                "z": tf_T[2]
            }
        }
        save_path = self.new_folder / 'arm_left_pose_in_world.json'
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

        # --------------------
        # arm_left_tcp_pose_in_world.json
        # --------------------
        tf = getTfTransform('base_link', 'gripper_left_base_link', returnMatrix=False)
        tf_T = tf[0]
        tf_Q = tf[1]
        roll, pitch, yaw = tft.euler_from_quaternion(tf_Q)
        data = {
            "rotation_euler_xyz_in_radians": {
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw
            },
            "position_in_meters": {
                "x": tf_T[0],
                "y": tf_T[1],
                "z": tf_T[2]
            }
        }
        save_path = self.new_folder / 'arm_left_tcp_pose_in_world.json'
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

        # --------------------
        # arm_right_joints.json
        # --------------------
        right_joints = self.joint_states.position[7:14]
        data = {
            "values": list(right_joints)
        }
        save_path = self.new_folder / 'arm_right_joints.json'
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)

        # --------------------
        # arm_right_pose_in_world.json
        # --------------------
        tf = getTfTransform('base_link', 'arm_right_1_link', returnMatrix=False)
        tf_T = tf[0]
        tf_Q = tf[1]
        roll, pitch, yaw = tft.euler_from_quaternion(tf_Q)
        data = {
            "rotation_euler_xyz_in_radians": {
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw
            },
            "position_in_meters": {
                "x": tf_T[0],
                "y": tf_T[1],
                "z": tf_T[2]
            }
        }
        save_path = self.new_folder / 'arm_right_pose_in_world.json'
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

        # --------------------
        # arm_right_tcp_pose_in_world.json
        # --------------------
        tf = getTfTransform('base_link', 'gripper_right_base_link', returnMatrix=False)
        tf_T = tf[0]
        tf_Q = tf[1]
        roll, pitch, yaw = tft.euler_from_quaternion(tf_Q)
        data = {
            "rotation_euler_xyz_in_radians": {
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw
            },
            "position_in_meters": {
                "x": tf_T[0],
                "y": tf_T[1],
                "z": tf_T[2]
            }
        }
        save_path = self.new_folder / 'arm_right_tcp_pose_in_world.json'
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

        # --------------------
        # camera_intrinsics.json
        # --------------------
        data = {
            "image_resolution": {
                "width": self.camera_info.width,
                "height": self.camera_info.height
            },
            "focal_lengths_in_pixels": {
                "fx": self.camera_info.K[0],
                "fy": self.camera_info.K[4]
            },
            "principal_point_in_pixels": {
                "cx": self.camera_info.K[2],
                "cy": self.camera_info.K[5]
            }
        }
        save_path = self.new_folder / 'camera_intrinsics.json'
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)

        # --------------------
        # camera_pose_in_world.json
        # --------------------
        tf = getTfTransform('base_link', 'rgbd_depth_optical_frame', returnMatrix=False)
        tf_T = tf[0]
        tf_Q = tf[1]
        roll, pitch, yaw = tft.euler_from_quaternion(tf_Q)
        data = {
            "rotation_euler_xyz_in_radians": {
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw
            },
            "position_in_meters": {
                "x": tf_T[0],
                "y": tf_T[1],
                "z": tf_T[2]
            }
        }
        save_path = self.new_folder / 'camera_pose_in_world.json'
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

        # --------------------
        # image_left.png
        # --------------------
        cv_img = self.bridge.imgmsg_to_cv2(self.rgb_img, desired_encoding='bgr8')
        save_path = self.new_folder / 'image_left.png'
        cv2.imwrite(str(save_path), cv_img)

        # --------------------
        # point_cloud.ply
        # --------------------
        points = np.array(
            list(pc2.read_points(self.points_msg, field_names=('x', 'y', 'z'), skip_nans=True)),
            dtype=np.float32
        )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        save_path = self.new_folder / 'point_cloud.ply'
        o3d.io.write_point_cloud(str(save_path), pcd)

        # --------------------
        # requested_model.json
        # --------------------
        data = {
            "requested_model": "v2+seg+rand_bg+cropping"
            #"requested_model": "v2+seg+rand_bg+cropping+score"
        }
        save_path = self.new_folder / 'requested_model.json'
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)

        self.cedirnet_trigger_pub.publish(True)

        rospy.loginfo('Data recording finished.')

        # --------------------
        # confidence_map.tiff
        # --------------------
        
        confidence_map = np.zeros((480, 640))
        save_path = self.new_folder / 'confidence_map.tiff'
        cv2.imwrite(str(save_path), confidence_map)

        # --------------------
        # depth_map.tiff
        # --------------------

        depth_map = np.zeros((480, 640))
        save_path = self.new_folder / 'depth_map.tiff'
        cv2.imwrite(str(save_path), depth_map)


if __name__ == '__main__':
    recorder = dataRecorder()
    rospy.spin()