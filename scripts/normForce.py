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
        self.pub_right_norm = rospy.Publisher(
            '/right_wrist_ft_norm',
            Float64,
            queue_size=1
        )

        self.pub_left_norm = rospy.Publisher(
            '/left_wrist_ft_norm',
            Float64,
            queue_size=1
        )

    def compute_norm(self, force):
        return math.sqrt(
            force.x ** 2 +
            force.y ** 2 +
            force.z ** 2
        )

    def right_ft_callback(self, msg):
        f = msg.wrench.force
        norm = self.compute_norm(f)

        out = Float64()
        out.data = norm

        self.pub_right_norm.publish(out)

    def left_ft_callback(self, msg):
        f = msg.wrench.force
        norm = self.compute_norm(f)

        out = Float64()
        out.data = norm

        self.pub_left_norm.publish(out)


if __name__ == '__main__':
    rospy.init_node('arms_force_norm_node')

    node = ArmsForceNormNode()
    rospy.loginfo('ArmsForceNormNode started.')

    rospy.spin()