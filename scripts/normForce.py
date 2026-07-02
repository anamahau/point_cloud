#!/usr/bin/env python3

import math
import rospy

from std_msgs.msg import Float64
from geometry_msgs.msg import WrenchStamped


class ArmsForceNormNode:
    def __init__(self):

        # Subscribers
        self.sub_right_ft = rospy.Subscriber(
            '/right_wrist_ft',
            WrenchStamped,
            self.right_ft_callback,
            queue_size=1
        )

        self.sub_left_ft = rospy.Subscriber(
            '/left_wrist_ft',
            WrenchStamped,
            self.left_ft_callback,
            queue_size=1
        )

        # Publishers
        self.pub_right_norm_force = rospy.Publisher(
            '/right_wrist_ft_norm_force',
            Float64,
            queue_size=1
        )

        self.pub_left_norm_force = rospy.Publisher(
            '/left_wrist_ft_norm_force',
            Float64,
            queue_size=1
        )

        self.pub_right_norm_torque = rospy.Publisher(
            '/right_wrist_ft_norm_torque',
            Float64,
            queue_size=1
        )

        self.pub_left_norm_torque = rospy.Publisher(
            '/left_wrist_ft_norm_torque',
            Float64,
            queue_size=1
        )

    def compute_norm_force(self, force):
        return math.sqrt(
            force.x ** 2 +
            force.y ** 2 +
            force.z ** 2
        )
    
    def compute_norm_torque(self, torque):
        return math.sqrt(
            torque.x ** 2 +
            torque.y ** 2 +
            torque.z ** 2
        )

    def right_ft_callback(self, msg):
        f = msg.wrench.force
        t = msg.wrench.torque
        norm_force = self.compute_norm_force(f)
        norm_torque = self.compute_norm_torque(t)

        out_force = Float64()
        out_torque = Float64()
        out_force.data = norm_force
        out_torque.data = norm_torque

        self.pub_right_norm_force.publish(out_force)
        self.pub_right_norm_torque.publish(out_torque)

    def left_ft_callback(self, msg):
        f = msg.wrench.force
        t = msg.wrench.torque
        norm_force = self.compute_norm_force(f)
        norm_torque = self.compute_norm_torque(t)

        out_force = Float64()
        out_torque = Float64()
        out_force.data = norm_force
        out_torque.data = norm_torque

        self.pub_left_norm_force.publish(out_force)
        self.pub_left_norm_torque.publish(out_torque)


if __name__ == '__main__':
    rospy.init_node('arms_force_norm_node')

    node = ArmsForceNormNode()
    rospy.loginfo('ArmsForceNormNode started.')

    rospy.spin()