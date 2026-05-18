#!/usr/bin/env python3

import rospy
import json
import os
import numpy as np
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import JointState, Image, PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tf_trans
import open3d as o3d

# ---- YOUR TF FUNCTION ----
# assumed available in your environment
from tf_reader import getTfTransform


class DataRecorder:

    def __init__(self):
        rospy.init_node("data_recorder_wait")

        self.bridge = CvBridge()
        self.save_path = rospy.get_param("~save_path", "/tmp/talos_data")
        os.makedirs(self.save_path, exist_ok=True)

    # ---------------- JSON ----------------
    def save_json(self, filename, data):
        with open(os.path.join(self.save_path, filename), 'w') as f:
            json.dump(data, f, indent=2)

    # ---------------- POSE ----------------
    def transform_to_dict(self, trans, quat):
        roll, pitch, yaw = tf_trans.euler_from_quaternion(quat)

        return {
            "rotation_euler_xyz_in_radians": {
                "yaw": yaw,
                "roll": roll,
                "pitch": pitch
            },
            "position_in_meters": {
                "x": trans[0],
                "y": trans[1],
                "z": trans[2]
            }
        }

    # ---------------- MAIN LOGIC ----------------
    def run(self):

        rospy.loginfo("Waiting for joint states...")
        joint_msg = rospy.wait_for_message("/joint_states", JointState)

        # ---- Extract joints by name (EDIT THESE NAMES) ----
        def get_joints(names):
            values = []
            for n in names:
                idx = joint_msg.name.index(n)
                values.append(joint_msg.position[idx])
            return values

        left_joint_names = [
            "arm_left_1_joint", "arm_left_2_joint", "arm_left_3_joint",
            "arm_left_4_joint", "arm_left_5_joint", "arm_left_6_joint",
            "arm_left_7_joint"
        ]

        right_joint_names = [
            "arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint",
            "arm_right_4_joint", "arm_right_5_joint", "arm_right_6_joint",
            "arm_right_7_joint"
        ]

        self.save_json("arm_left_joints.json",
                       {"values": get_joints(left_joint_names)})

        self.save_json("arm_right_joints.json",
                       {"values": get_joints(right_joint_names)})

        # ---- CAMERA INFO ----
        rospy.loginfo("Waiting for camera info...")
        cam_info = rospy.wait_for_message("/rgbd/depth/camera_info", CameraInfo)

        cam_intrinsics = {
            "image_resolution": {
                "width": cam_info.width,
                "height": cam_info.height
            },
            "focal_lengths_in_pixels": {
                "fx": cam_info.K[0],
                "fy": cam_info.K[4]
            },
            "principal_point_in_pixels": {
                "cx": cam_info.K[2],
                "cy": cam_info.K[5]
            }
        }
        self.save_json("camera_intrinsics.json", cam_intrinsics)

        # ---- IMAGES ----
        rospy.loginfo("Waiting for images...")
        img_msg = rospy.wait_for_message("/rgbd/color/image", Image)
        # depth_msg = rospy.wait_for_message("/rgbd/depth/image", Image)
        # conf_msg = rospy.wait_for_message("/rgbd/confidence", Image)

        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        # depth = self.bridge.imgmsg_to_cv2(depth_msg)
        # conf = self.bridge.imgmsg_to_cv2(conf_msg)

        cv2.imwrite(os.path.join(self.save_path, "image_left.png"), img)
        # cv2.imwrite(os.path.join(self.save_path, "depth_map.tiff"), depth)
        # cv2.imwrite(os.path.join(self.save_path, "confidence_map.tiff"), conf)

        # ---- POINT CLOUD ----
        rospy.loginfo("Waiting for point cloud...")
        pc_msg = rospy.wait_for_message("/rgbd/points", PointCloud2)

        points = []
        for p in pc2.read_points(pc_msg, skip_nans=True):
            points.append([p[0], p[1], p[2]])

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.array(points))

        o3d.io.write_point_cloud(
            os.path.join(self.save_path, "point_cloud.ply"), pc
        )

        # ---- TF TRANSFORMS ----
        rospy.loginfo("Getting TF transforms...")

        def get_pose(child, parent="base_link"):
            trans, quat = getTfTransform(child, parent, returnMatrix=False)
            return self.transform_to_dict(trans, quat)

        self.save_json("camera_pose_in_world.json",
                       get_pose("head_2_link"))

        self.save_json("arm_left_pose_in_world.json",
                       get_pose("arm_left_7_link"))

        self.save_json("arm_right_pose_in_world.json",
                       get_pose("arm_right_7_link"))

        self.save_json("arm_left_tcp_pose_in_world.json",
                       get_pose("arm_left_tool_link"))

        self.save_json("arm_right_tcp_pose_in_world.json",
                       get_pose("arm_right_tool_link"))

        # ---- STATIC ----
        self.save_json("requested_model.json", {"requested_model": "v2"})

        rospy.loginfo("Data saved successfully.")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    recorder = DataRecorder()
    recorder.run()