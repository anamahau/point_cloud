#!/usr/bin/env python

import cv2 as cv
import numpy as np
import open3d as o3d

import copy
import math
import rospy
import struct
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from point_cloud.msg import highLowPoint
from std_msgs.msg import Bool
from geometry_msgs.msg import Polygon, Point32

from tf_reader import getTfTransform


def pointcloud2_to_xyz_array(cloud_msg, maxZ=math.inf):
    '''
    Convert a ROS PointCloud2 message to a Nx3 NumPy array
    '''
    points = []
    # jointValue = 0.78
    jointValue = 0
    R = np.array([[ math.cos(jointValue), 0, math.sin(jointValue)],
                  [                    0, 1,                    0],
                  [-math.sin(jointValue), 1, math.cos(jointValue)]])
    # R_tf = getTfTransform('base_link', 'head_2_link')
    R_tf = getTfTransform('odom', 'head_2_link')
    # R = np.eye(4)
    R_rot = np.array([
        [ 0,  0, 1, 0],
        [-1,  0, 0, 0],
        [ 0, -1, 0, 0],
        [ 0,  0, 0, 1]
    ])
    R_rot2 = np.array([
        [0,  0, 1, 0],
        [0, -1, 0, 0],
        [1,  0, 0, 0],
        [0,  0, 0, 1]
    ])
    R = R_rot2 @ (R_tf @ R_rot)
    print('========== R matrix:')
    print(R)
    for p in pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True):
        if p[2] < maxZ:
            R_xyz = R @ np.array([p[0], p[1], p[2], 1])
            points.append([R_xyz[0], R_xyz[1], R_xyz[2]])
    return np.array(points)

def create_open3d_cloud(xyz_array, rgb_array):
    '''
    Convert Nx3 NumPy array to Open3D PointCloud
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_array)
    pcd.colors = o3d.utility.Vector3dVector(rgb_array)
    return pcd

def color_pc(pc, indexes, color):
    updated_colors = np.array(pc.colors)
    updated_colors[indexes] = color
    pc.colors = o3d.utility.Vector3dVector(updated_colors)
    return pc

def remove_points(pc, indexes):
    cuttedPC = o3d.geometry.PointCloud()
    xyz_new = []
    rgb_new = []
    for i in range(len(pc.points)):
        if indexes[i]:
            xyz_new.append(pc.points[i])
            rgb_new.append(pc.colors[i])
    cuttedPC.points = o3d.utility.Vector3dVector(xyz_new)
    cuttedPC.colors = o3d.utility.Vector3dVector(rgb_new)
    return cuttedPC

def findLowHigh(pc):
    points = np.asarray(pc.points)
    colors = np.asarray(pc.colors)
    z = points[:, 2]
    z_low_thresh = np.percentile(z, 1)    # bottom 1%
    z_high_thresh = np.percentile(z, 99)  # top 1%
    lowest_part = points[z <= z_low_thresh]
    highest_part = points[z >= z_high_thresh]
    print('========== highest point:')
    print(np.mean(highest_part, axis=0))
    print('========== lowest point:')
    print(np.mean(lowest_part, axis=0))
    colors[z <= z_low_thresh] = [0, 0, 1]   # blue = lowest
    colors[z >= z_high_thresh] = [0, 1, 0]  # green = highest
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc, np.mean(highest_part, axis=0), np.mean(lowest_part, axis=0)

def publishLowHigh(low, high):
    print('========== publisher 1')
    pub = rospy.Publisher('/highLowPCpoints', highLowPoint, queue_size=10)
    msg = highLowPoint()
    pub.publish(msg)
    rospy.sleep(0.5)
    print('========== publisher 2')



if __name__ == '__main__':

    rospy.init_node('pointCloud', anonymous=True)
    pub_old = rospy.Publisher('/highLowPCpoints', highLowPoint, queue_size=1, latch=True)

    print(':)')

    pc = rospy.wait_for_message('/rosbag/points', PointCloud2)
    print('PointCloud received')
    # pc = rospy.wait_for_message('/camera/depth_registered/points', PointCloud2)
    # rospy.loginfo('PointCloud2 received: width=%d height=%d' %(pc.width, pc.height))

    # rospy.loginfo('Number of points: %d' %(len(pc.data)/pc.point_step))

    # xyz_array = pointcloud2_to_xyz_array(pc)
    # rgb_array = np.zeros_like(xyz_array)
    # o3dPC = create_open3d_cloud(xyz_array, rgb_array)
    # o3d.visualization.draw_geometries([o3dPC], 'Point Cloud')

    xyz_array_2 = pointcloud2_to_xyz_array(pc, maxZ=2)
    rgb_array_2 = np.zeros_like(xyz_array_2)
    o3dPC_2 = create_open3d_cloud(xyz_array_2, rgb_array_2)
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=o3dPC_2.get_center())
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # o3d.visualization.draw_geometries([o3dPC_2, axis], 'Cropped Point Cloud')

    if True:
        # x = pc.data[:4]
        # y = pc.data[4:8]
        # z = pc.data[8:12]
        # print('point x:', struct.unpack('<f', x)[0])
        # print('point y:', struct.unpack('<f', y)[0])
        # print('point z:', struct.unpack('<f', z)[0])

        plane_model, inliers = o3dPC_2.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
        o3dPC_2 = color_pc(o3dPC_2, inliers, np.array([1, 0.6, 0]))
        print('plane model:', plane_model)
        o3d.visualization.draw_geometries([o3dPC_2], 'Plane segmentation')

        points = np.array(o3dPC_2.points)
        mask_plane = np.zeros(len(points), dtype=bool)
        mask_plane[inliers] = True
        a, b, c, d = plane_model
        mask_above = (a*points[:,0] + b*points[:,1] + c*points[:,2] + d > 0)
        indexesUp = mask_above & ~mask_plane
        
        twoColorPC = color_pc(o3dPC_2, indexesUp, np.array([1, 0, 1]))
        o3d.visualization.draw_geometries([twoColorPC, axis], '2 colors PC')

        objectsPC = remove_points(twoColorPC, indexesUp)
        o3d.visualization.draw_geometries([objectsPC], 'Objects')

        LHpc, highPC, lowPC = findLowHigh(objectsPC)
        o3d.visualization.draw_geometries([LHpc, axis], '2 points')
    
    print('========== publisher 1')
    pub = rospy.Publisher('/highLowPCpoints_new', Polygon, queue_size=10, latch=True)
    while rospy.Time.now().to_sec() == 0:
        rospy.sleep(0.1)
    rospy.sleep(0.5)
    msg_old = highLowPoint()
    msg = Polygon()
    pH = Point32()
    pL = Point32()
    pH.x = highPC[0]
    pH.y = highPC[1]
    pH.z = highPC[2]
    pL.x = lowPC[0]
    pL.y = lowPC[1]
    pL.z = lowPC[2]
    msg.points.append(pH)
    msg.points.append(pL)
    pub.publish(msg)
    # msg.highestPoint = highPC
    # msg.lowestPoint = lowPC
    # pub.publish(msg)
    print('========== publisher 2')

    '''
    point cloud je zapisan kot 2D array, kjer ima vsaka tocka 6 vrednosti [x, y, z, r, g, b]
    r, g, in b so normirane vrednosti
    '''