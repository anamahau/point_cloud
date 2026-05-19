#!/usr/bin/env python

import cv2 as cv
import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import copy
import math
import rospy
import struct
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from point_cloud.msg import highLowPoint
from std_msgs.msg import Bool, Int32, Float32MultiArray, Float32
from geometry_msgs.msg import Polygon, Point32

from tf_reader import getTfTransform


def save_point_cloud_plot(o3d_pc, output_path='point_cloud.png',
                          title='Point Cloud',
                          point_size=1,
                          dpi=300):
    '''
    Save an Open3D point cloud as a Matplotlib 3D image.

    Parameters:
    - o3d_pc: open3d.geometry.PointCloud
    - output_path: file path to save image
    - title: plot title
    - point_size: scatter point size
    - dpi: image resolution
    '''

    points = np.asarray(o3d_pc.points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Use colors if available
    if o3d_pc.has_colors():
        colors = np.asarray(o3d_pc.colors)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=colors, s=point_size)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   s=point_size)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev=30, azim=45)
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)  # Important: prevents displaying window & frees memory


def pointcloud2_to_xyz_array(cloud_msg, iteration, maxZ=math.inf):
    '''
    Convert a ROS PointCloud2 message to a Nx3 NumPy array
    '''
    print('Hello from pointcloud2_to_xyz_array :)')
    print('iteration ', iteration)
    points = []
    R = getTfTransform('base_link', 'rgbd_depth_optical_frame')
    print('========== R matrix:')
    print(R)
    groundZ = -0.9
    if iteration == 1:
        for p in pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True):
            R_xyz = R @ np.array([p[0], p[1], p[2], 1])
            if (R_xyz[2] < maxZ and R_xyz[2] > groundZ):
                points.append([R_xyz[0], R_xyz[1], R_xyz[2]])
    elif iteration == 2:
        for p in pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True):
            R_xyz = R @ np.array([p[0], p[1], p[2], 1])
            # if (R_xyz[2] < maxZ and R_xyz[1] < -0.1):
            if (R_xyz[0] < 0.8 and R_xyz[2] < 2.0 and R_xyz[2] > -0.1):
                points.append([R_xyz[0], R_xyz[1], R_xyz[2]])
    elif iteration == 3:
        for p in pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True):
            R_xyz = R @ np.array([p[0], p[1], p[2], 1])
            if (R_xyz[2] < maxZ and R_xyz[2] > groundZ):
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
    z_low_thresh = np.percentile(z, 1) # bottom 1%
    z_high_thresh = np.percentile(z, 99) # top 1%
    lowest_part = points[z <= z_low_thresh]
    highest_part = points[z >= z_high_thresh]
    print('========== highest point:')
    print(np.mean(highest_part, axis=0))
    print('========== lowest point:')
    print(np.mean(lowest_part, axis=0))
    colors[z <= z_low_thresh] = [0, 0, 1] # blue = lowest
    colors[z >= z_high_thresh] = [0, 1, 0] # green = highest
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc, np.mean(highest_part, axis=0), np.mean(lowest_part, axis=0)

def publishLowHigh(low, high):
    print('========== publisher 1')
    pub = rospy.Publisher('/highLowPCpoints', highLowPoint, queue_size=10)
    msg = highLowPoint()
    pub.publish(msg)
    rospy.sleep(0.5)
    print('========== publisher 2')

run_analysis = False
iteration_data = 0

def trigger_cb(msg):
    print('Hello from trigger_cb :)')
    global run_analysis
    global iteration_data
    run_analysis = True
    iteration_data = msg.data

def PCanalysis(iteration):
    print('Hello from PCanalysis :)')
    print('number ', iteration)
    pc = rospy.wait_for_message('/rgbd/depth/points', PointCloud2)

    xyz_array_2 = pointcloud2_to_xyz_array(pc, iteration=iteration, maxZ=2)
    rgb_array_2 = np.zeros_like(xyz_array_2)
    o3dPC_2 = create_open3d_cloud(xyz_array_2, rgb_array_2)
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=o3dPC_2.get_center())
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([o3dPC_2, axis], 'Cropped Point Cloud')

    if iteration == 1:

        plane_model, inliers = o3dPC_2.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        o3dPC_2 = color_pc(o3dPC_2, inliers, np.array([1, 0.6, 0]))
        print('plane model:', plane_model)
        # o3d.visualization.draw_geometries([o3dPC_2], 'Plane segmentation')
        save_point_cloud_plot(o3dPC_2, 
                      output_path='plane_segmentation.png',
                      title='Plane Segmentation')

        points = np.array(o3dPC_2.points)
        mask_plane = np.zeros(len(points), dtype=bool)
        mask_plane[inliers] = True
        a, b, c, d = plane_model
        mask_above = (a*points[:,0] + b*points[:,1] + c*points[:,2] + d > 0)
        indexesUp = mask_above & ~mask_plane
        
        twoColorPC = color_pc(o3dPC_2, indexesUp, np.array([1, 0, 1]))
        # o3d.visualization.draw_geometries([twoColorPC, axis], '2 colors PC')
        save_point_cloud_plot(twoColorPC, 
                      output_path='2_colors_PC.png',
                      title='2 colors PC')

        objectsPC = remove_points(twoColorPC, indexesUp)
        # o3d.visualization.draw_geometries([objectsPC], 'Objects')
        save_point_cloud_plot(objectsPC, 
                      output_path='Objects.png',
                      title='Objects')

        LHpc, highPC, lowPC = findLowHigh(objectsPC)
        # o3d.visualization.draw_geometries([LHpc, axis], '2 points')
        save_point_cloud_plot(LHpc, 
                      output_path='2 points.png',
                      title='2 points')
    
    elif iteration == 2:
        LHpc, highPC, lowPC = findLowHigh(o3dPC_2)
        save_point_cloud_plot(LHpc, 
                      output_path='2 points.png',
                      title='2 points')
        
    elif iteration == 3:
        plane_model, inliers = o3dPC_2.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        print('plane model:', plane_model)
        plane_points = np.asarray(o3dPC_2.points)[inliers]
        table_height = np.mean(plane_points[:, 2])
        print("Estimated table height (z):", table_height)
        o3dPC_2 = color_pc(o3dPC_2, inliers, np.array([1, 0.6, 0]))
        o3d.visualization.draw_geometries([o3dPC_2], 'Plane segmentation')
    
    # print('========== publisher 1')
    if iteration == 1:
        pub = rospy.Publisher('/highPCpoint', Float32MultiArray, queue_size=10, latch=True)
        while rospy.Time.now().to_sec() == 0:
            rospy.sleep(0.1)
        rospy.sleep(0.5)
        hPC = Float32MultiArray()
        hPC.data = highPC
        pub.publish(hPC)
    elif iteration == 2:
        pub = rospy.Publisher('/lowPCpoint', Float32MultiArray, queue_size=10, latch=True)
        while rospy.Time.now().to_sec() == 0:
            rospy.sleep(0.1)
        rospy.sleep(0.5)
        lPC = Float32MultiArray()
        lPC.data = lowPC
        pub.publish(lPC)
    elif iteration == 3:
        pub = rospy.Publisher('/heightOfTable', Float32, queue_size=10, latch=True)
        while rospy.Time.now().to_sec() == 0:
            rospy.sleep(0.1)
        rospy.sleep(0.5)
        hot = Float32()
        hot.data = table_height
        pub.publish(hot)
    # pub = rospy.Publisher('/highLowPCpoints_new', Polygon, queue_size=10, latch=True)
    # while rospy.Time.now().to_sec() == 0:
    #     rospy.sleep(0.1)
    # rospy.sleep(0.5)
    # msg_old = highLowPoint()
    # msg = Polygon()
    # pH = Point32()
    # pL = Point32()
    # pH.x = highPC[0]
    # pH.y = highPC[1]
    # pH.z = highPC[2]
    # pL.x = lowPC[0]
    # pL.y = lowPC[1]
    # pL.z = lowPC[2]
    # msg.points.append(pH)
    # msg.points.append(pL)
    # pub.publish(msg)
    # msg.highestPoint = highPC
    # msg.lowestPoint = lowPC
    # pub.publish(msg)
    # print('========== publisher 2')



if __name__ == '__main__':

    rospy.init_node('pointCloud', anonymous=True)
    pub_old = rospy.Publisher('/highLowPCpoints', highLowPoint, queue_size=1, latch=True)

    rospy.Subscriber('/PCrequest', Int32, trigger_cb)

    # for i in range(2):
    # for i in range(1):
    #     run_analysis = 0

    #     rate = rospy.Rate(10)
    #     while not rospy.is_shutdown() and not run_analysis:
    #         rate.sleep()

    #     PCanalysis(run_analysis)

    rate = rospy.Rate(10)
    # while not rospy.is_shutdown() and not run_analysis:
    #     rate.sleep()
    # PCanalysis(iteration_data)
    while not rospy.is_shutdown():
        if run_analysis:
            print('Trigger received, running analysis...')
            PCanalysis(iteration_data)
            print('Point cloud analysis is done!')
            run_analysis = False
        rate.sleep()

    '''
    point cloud je zapisan kot 2D array, kjer ima vsaka tocka 6 vrednosti [x, y, z, r, g, b]
    r, g, in b so normirane vrednosti
    '''