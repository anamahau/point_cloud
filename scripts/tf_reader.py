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


tf_buffer = None
tf_listener = None

def init_tf():
    global tf_buffer, tf_listener
    tf_buffer = tf2_ros.Buffer(
        cache_time=rospy.Duration(30.0)
    )
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(1.0)


def getTfTransform(firstLink, secondLink, returnMatrix=True):
    global tf_buffer

    try:
        current_transform = tf_buffer.lookup_transform(
            firstLink,
            secondLink,
            rospy.Time(0),
            rospy.Duration(1.0)
        )
        translation = current_transform.transform.translation
        rotation = current_transform.transform.rotation
        if returnMatrix:
            return quaternion2rotationMatrix(current_transform.transform)
        return [[translation.x, translation.y, translation.z], [rotation.x, rotation.y, rotation.z, rotation.w]]
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException
    ) as e:
        rospy.logwarn(
            "TF lookup failed (%s -> %s): %s",
            firstLink,
            secondLink,
            str(e)
        )
        return None


if __name__ == '__main__':

    rospy.init_node('tf_reader', anonymous=True)

    init_tf()

    # transform = getTfTransform('head_2_link', 'odom')
    # print(transform)
    # transform = getTfTransform('head_2_link', 'odom', returnMatrix=False)
    # print(transform)

    # print('rgbd_depth_optical_frame:')
    # print(getTfTransform('odom', 'rgbd_depth_optical_frame'))
    # print(getTfTransform('odom', 'rgbd_depth_optical_frame', returnMatrix=False))

    print('gripper_right_base_link:')
    print(getTfTransform('odom', 'gripper_right_base_link'))
    print(getTfTransform('odom', 'gripper_right_base_link', returnMatrix=False))

    print('gripper_left_base_link:')
    print(getTfTransform('odom', 'gripper_left_base_link'))
    print(getTfTransform('odom', 'gripper_left_base_link', returnMatrix=False))

    # print('arm_right_1_link:')
    # print(getTfTransform('odom', 'arm_right_1_link'))
    # print(getTfTransform('odom', 'arm_right_1_link', returnMatrix=False))

    # print('arm_left_1_link:')
    # print(getTfTransform('odom', 'arm_left_1_link'))
    # print(getTfTransform('odom', 'arm_left_1_link', returnMatrix=False))

    # print('arm_right_7_link:')
    # print(getTfTransform('odom', 'arm_right_7_link'))
    # print(getTfTransform('odom', 'arm_right_7_link', returnMatrix=False))

    # print('arm_left_7_link:')
    # print(getTfTransform('odom', 'arm_left_7_link'))
    # print(getTfTransform('odom', 'arm_left_7_link', returnMatrix=False))