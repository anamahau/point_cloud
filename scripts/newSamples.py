#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Bool, Int32


def jointsMove(joints):
    pub = rospy.Publisher(
        '/head_controller/command',
        JointTrajectory,
        queue_size=10
    )

    rospy.sleep(1)

    traj = JointTrajectory()

    traj.joint_names = ["head_1_joint", "head_2_joint"]

    point = JointTrajectoryPoint()
    point.positions = joints
    point.velocities = [0.0, 0.0]
    point.time_from_start = rospy.Duration(5.0)

    traj.points = [point]

    traj.header.stamp = rospy.Time.now()
    pub.publish(traj)


def cedirnetMove():
    headPositions = [
        # [0.28, 0.0],
        # [0.53, 0.0],
        [0.38, 0.0],
        [0.58, 0.0],
        [0.78, 0.0]
    ]
    
    triger_pub = rospy.Publisher('/data_recorder/trigger2', Int32, queue_size=10)
    i = 1
    for joints in headPositions:
        print(joints)
        CEDIRNET_FINISHED = False
        jointsMove(joints)
        rospy.sleep(5.0)
        print('move done')
        msg = Int32()
        msg.data = i
        triger_pub.publish(msg)
        i += 1
        rate = rospy.Rate(10)
        finished = rospy.wait_for_message('/cedirnet/finished', Bool)
        # while (not CEDIRNET_FINISHED):
        #     print('while', CEDIRNET_FINISHED)
        #     # rospy.sleep(0.0)
        #     rate.sleep()
        print('end of while loop')
    jointsMove([0.2, 0.0])


def callback(msg):
    global CEDIRNET_FINISHED
    CEDIRNET_FINISHED = msg.data
    print('callback:', CEDIRNET_FINISHED)


if __name__ == '__main__':
    rospy.init_node('new_smaples_node', anonymous=True)
    finished_sub = rospy.Subscriber('/cedirnet/finished', Bool, callback)

    cedirnetMove()