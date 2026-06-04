#!/usr/bin/env python3

import cv2
import glob
import yaml
import json
import rospy
import tf2_ros
import argparse
# import tf2_geometry_msgs

import numpy as np

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from tf.transformations import quaternion_from_matrix, rotation_matrix

from scipy.spatial.transform import Rotation as scipyR

CHECKERBOARD = (10, 5)
SQUARE_SIZE = 0.0463

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

with open('./src/point_cloud/scripts/camera_color_cameraInfo.yaml', 'r') as stream:
    calib_data = yaml.safe_load(stream)
mtx_rs = np.array(calib_data['camera_matrix']['data']).reshape(3,3)
dist_rs = np.array(calib_data['distortion_coefficients']['data'])

with open('./src/point_cloud/scripts/rgbd_rgb_cameraInfo.yaml', 'r') as stream:
    calib_data = yaml.safe_load(stream)
mtx_orb = np.array(calib_data['camera_matrix']['data']).reshape(3,3)
dist_orb = np.array(calib_data['distortion_coefficients']['data'])


class FindRealSenseTF():

    def __init__(self):

        self.color_sub_orb_ = rospy.Subscriber('/rgbd/rgb/image_raw', Image, self.color_cb_orb_)
        self.color_sub_rs_ = rospy.Subscriber('/camera/color/image_raw', Image, self.color_cb_rs_)
        self.bridge = CvBridge()
        self.color_msg_orb_ = None
        self.color_msg_rs_ = None

        self.rvec_orb = None
        self.tvec_orb = None
        self.rvec_rs = None
        self.tvec_rs = None
        self.corners_orb = None
        self.corners_rs = None
        self.elapsed_time = None
        self.C_DIFF = None
        self.P_DIFF = None
        self.R_DIFF = np.eye(4)

        while (self.color_msg_orb_ is None or self.color_msg_rs_ is None) and not rospy.is_shutdown():
            rospy.sleep(0.1)
        print(':)')
        self.detectChessboard()


    def color_cb_orb_(self, data):
        self.color_msg_orb_ = data
        self.color_img_orb_ = self.bridge.imgmsg_to_cv2(data, 'bgr8')

    def color_cb_rs_(self, data):
        self.color_msg_rs_ = data
        self.color_img_rs_ = self.bridge.imgmsg_to_cv2(data, 'bgr8')

    def saveImage(self, image, path):
        cv2.imwrite(path, image)

    def detectChessboard(self):
        gray_orb = cv2.cvtColor(self.color_img_orb_, cv2.COLOR_BGR2GRAY)
        gray_rs = cv2.cvtColor(self.color_img_rs_, cv2.COLOR_BGR2GRAY)

        ret, corners_rs = cv2.findChessboardCornersSB(gray_rs, CHECKERBOARD, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        corners_img_rs = self.color_img_rs_.copy()
        cv2.drawChessboardCorners(corners_img_rs, CHECKERBOARD, corners_rs, ret)
        self.saveImage(corners_img_rs, './RScallibration/corners_img_rs.png')
        
        self.corners_rs = cv2.cornerSubPix(gray_rs, corners_rs, (11,11), (-1,-1), criteria)
        ret, self.rvec_rs, self.tvec_rs = cv2.solvePnP(objp, corners_rs, mtx_rs, dist_rs)
        R = cv2.Rodrigues(self.rvec_rs)[0]
        self.C_RS = np.array([R[:, 0], R[:, 1], R[:, 2], self.tvec_rs.squeeze()]).T
        self.C_RS = np.concatenate((self.C_RS, np.array([[0, 0, 0, 1]])), axis=0)

        ret, corners_orb = cv2.findChessboardCornersSB(gray_orb, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        corners_img_orb = self.color_img_orb_.copy()
        cv2.drawChessboardCorners(corners_img_orb, CHECKERBOARD, corners_orb, ret)
        self.saveImage(corners_img_orb, './RScallibration/corners_img_orb.png')

        self.corners_orb = cv2.cornerSubPix(gray_orb, corners_orb, (11,11), (-1,-1), criteria)
        ret, self.rvec_orb, self.tvec_orb = cv2.solvePnP(objp, corners_orb, mtx_orb, dist_orb)
        R = cv2.Rodrigues(self.rvec_orb)[0]
        self.C_ORB = np.array([R[:, 0], R[:, 1], R[:, 2], self.tvec_orb.squeeze()]).T
        self.C_ORB = np.concatenate((self.C_ORB, np.array([[0, 0, 0, 1]])), axis=0)

        self.C_DIFF = np.matmul(self.C_ORB, np.linalg.inv(self.C_RS))
        self.P_DIFF = (self.tvec_orb - self.tvec_rs)[:,-1]
        self.R_DIFF[0:3,0:3] = np.matmul(np.linalg.inv(self.C_RS[0:3,0:3]), self.C_ORB[0:3,0:3])

        # HACK: apparently this works better than using P_DIFF and R_DIFF
        self.P_DIFF = self.C_DIFF[0:3,-1]
        self.R_DIFF[0:3,0:3] = self.C_DIFF[0:3,0:3]

        print('######\nFound transform!\n######')
        print(self.P_DIFF)
        print(self.R_DIFF)

        r = scipyR.from_matrix(self.R_DIFF[:3, :3])
        print(r.as_euler('xyz', degrees=False), 'rad')
        print(r.as_euler('xyz', degrees=True), 'deg')
        print(r.as_quat())


if __name__ == '__main__':

    rospy.init_node('findRealSenseTF_node', anonymous=True)

    findRsTF = FindRealSenseTF()



'''
https://www.andre-gaschler.com/rotationconverter/

[-0.00320142 -0.10964579 -0.05616454] :)
[[ 0.99915348  0.04039518  0.00778156  0.        ]
 [-0.04021681  0.99895189 -0.02185725  0.        ]
 [-0.00865633  0.0215258   0.99973082  0.        ]
 [ 0.          0.          0.          1.        ]]
[ x: 0.0218597, y: 0.0077816, z: -0.0404074 ] rad
[ x: 1.2524659, y: 0.445855, z: -2.3151736 ] deg
[ 0.0108487, 0.0041106, -0.0201585, 0.9997295 ]

[-0.01869195 -0.01932996 -0.06425512]
[[ 0.9990264   0.04030072  0.01794722  0.        ]
 [-0.03874562  0.99605098 -0.07988261  0.        ]
 [-0.02109568  0.07910946  0.9966427   0.        ]
 [ 0.          0.          0.          1.        ]]
[ x: 0.0799807, y: 0.0179482, z: -0.0403181 ] rad
[ x: 4.5825576, y: 1.0283553, z: -2.3100589 ] deg
[ 0.0397892, 0.0097708, -0.0197821, 0.9989645 ]

[-0.02546587 -0.11541605 -0.0493693 ]
[[ 0.9989575   0.04022338  0.02158668  0.        ]
 [-0.03988856  0.99908041 -0.0157236   0.        ]
 [-0.02219929  0.01484615  0.99964333  0.        ]
 [ 0.          0.          0.          1.        ]]
[ x: 0.0157279, y: 0.0215884, z: -0.0402436 ] rad
[ x: 0.9011431, y: 1.2369219, z: -2.3057895 ] deg :)
[ 0.0076447, 0.0109497, -0.0200338, 0.9997101 ]

[-0.00777354 -0.122649   -0.05410189]
[[ 0.99913153  0.04053211  0.00966113  0.        ]
 [-0.04042995  0.99912673 -0.01054518  0.        ]
 [-0.01008011  0.01014542  0.99989773  0.        ]
 [ 0.          0.          0.          1.        ]]
[ x: 0.0105459, y: 0.0096613, z: -0.0405451 ] rad
[ x: 0.6042337, y: 0.5535506, z: -2.3230636 ] deg
[ 0.0051738, 0.0049364, -0.0202452, 0.9997695 ]
'''