#!/usr/bin/env python

import copy
import math
import rospy
import struct
import tf2_ros

import numpy as np

from geometry_msgs.msg import Pose


def quaternion2rotationMatrix(transform):
    """
    https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)
    q0 = x, q1 = y, q2 = z, q3 = w
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """

    x = transform.rotation.x
    y = transform.rotation.y
    z = transform.rotation.z
    w = transform.rotation.w

    r00 = 1 - 2*(y*y + z*z)
    r01 = 2*(x*y - z*w)
    r02 = 2*(x*z + y*w)

    r10 = 2*(x*y + z*w)
    r11 = 1 - 2*(x*x + z*z)
    r12 = 2*(y*z - x*w)

    r20 = 2*(x*z - y*w)
    r21 = 2*(y*z + x*w)
    r22 = 1 - 2*(x*x + y*y)

    tx = transform.translation.x
    ty = transform.translation.y
    tz = transform.translation.z

    matrix = np.array([
        [r00, r01, r02, tx],
        [r10, r11, r12, ty],
        [r20, r21, r22, tz],
        [  0,   0,   0,  1]
    ])


    return matrix


def getTfTransform(firstLink, secondLink, returnMatrix=True):
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(0.5)
    current_transform = tf_buffer.lookup_transform(
        firstLink,
        secondLink,
        rospy.Time(0),
        rospy.Duration(1.0)# ,
        # rospy.Duration(1.0)
    )
    translation = current_transform.transform.translation
    rotation = current_transform.transform.rotation
    if returnMatrix:
        return quaternion2rotationMatrix(current_transform.transform)
    return [[translation.x, translation.y, translation.z], [rotation.x, rotation.y, rotation.z, rotation.w]]


if __name__ == '__main__':

    rospy.init_node('tf_reader', anonymous=True)

    # transform = getTfTransform('head_2_link', 'base_link')
    # print(transform)
    # transform = getTfTransform('head_2_link', 'base_link', returnMatrix=False)
    # print(transform)

    # print('rgbd_depth_optical_frame:')
    # print(getTfTransform('base_link', 'rgbd_depth_optical_frame'))
    # print(getTfTransform('base_link', 'rgbd_depth_optical_frame', returnMatrix=False))

    print('gripper_right_base_link:')
    print(getTfTransform('base_link', 'gripper_right_base_link'))
    print(getTfTransform('base_link', 'gripper_right_base_link', returnMatrix=False))

    print('gripper_left_base_link:')
    print(getTfTransform('base_link', 'gripper_left_base_link'))
    print(getTfTransform('base_link', 'gripper_left_base_link', returnMatrix=False))

    # print('arm_right_1_link:')
    # print(getTfTransform('base_link', 'arm_right_1_link'))
    # print(getTfTransform('base_link', 'arm_right_1_link', returnMatrix=False))

    # print('arm_left_1_link:')
    # print(getTfTransform('base_link', 'arm_left_1_link'))
    # print(getTfTransform('base_link', 'arm_left_1_link', returnMatrix=False))

    # print('arm_right_7_link:')
    # print(getTfTransform('base_link', 'arm_right_7_link'))
    # print(getTfTransform('base_link', 'arm_right_7_link', returnMatrix=False))

    # print('arm_left_7_link:')
    # print(getTfTransform('base_link', 'arm_left_7_link'))
    # print(getTfTransform('base_link', 'arm_left_7_link', returnMatrix=False))