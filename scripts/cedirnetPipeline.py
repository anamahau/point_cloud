#!/usr/bin/env python3
# cedirnet_pipeline.py
#
# Replaces saveData.py + sshBridge_test.py. Keeps panorama_image.py (imported).
#
# Flow (trigger values come from headMove::cedirnetMove on /data_recorder/trigger2):
#   1 -> new sample, save image_1.png / depth_1.tiff
#   2 -> save image_2.png / depth_2.tiff + ALL other data (middle head position)
#   3 -> save image_3.png / depth_3.tiff, build panorama -> image_left.png / depth_map.tiff,
#        copy the ONE sample to the remote machine, wait for CeDiRNet, publish /cedirnet/goal_pose
# After every step /cedirnet/finished (Bool) is published, also on errors.

import re
import json
import subprocess
from pathlib import Path

import cv2
import rospy
import numpy as np
import tf.transformations as tft

from PIL import Image as ImagePIL
from cv_bridge import CvBridge
from tf_reader import getTfTransform, init_tf
from std_msgs.msg import Bool, Int32
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState, Image, CameraInfo

from panorama_image import stitch_three

# =========================
# CONFIG
# =========================
BASE_DIR = Path('/home/pal/docker_anamarija/dataForCedirnet')
REMOTE_USER = 'anamarija'
REMOTE_HOST = '178.172.42.49'
REMOTE_PATH = '/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference'
RESULT_TIMEOUT = 300.0  # s

TRIGGER_TOPIC = '/data_recorder/trigger2'
# RGB_TOPIC = '/camera2/camera/color/image_raw'
RGB_TOPIC = '/rgbd/rgb/image_raw'
# DEPTH_TOPIC = '/camera2/camera/aligned_depth_to_color/image_raw'
DEPTH_TOPIC = '/rgbd/depth/image_raw'
# INFO_TOPIC = '/camera2/camera/color/camera_info'
INFO_TOPIC = '/rgbd/rgb/camera_info'

REQUESTED_MODEL = 'v2+seg+rand_bg+cropping_talos'

# Panorama geometry. The 192/272 offsets are hard-coded in stitch_three(); the rest is the
# resize/pad you used in generatePanoramaSample(). Final image: 1024x1024.
PAN_OFFSET_X, PAN_OFFSET_Y = 192, 272
PAN_SIZE = 768
PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT = 0, 256, 128, 128
FINAL_SIZE = 1024

# False = keep the middle camera's fx, fy, cx, cy (what you did so far).
# True  = transform them to match the stitched, resized, padded image.
ADJUST_INTRINSICS = False


def to_meters(msg, bridge):
    depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    if depth.dtype == np.uint16:        # mm -> m
        depth = depth.astype(np.float32) / 1000.0
    return depth.astype(np.float32)


class CedirnetPipeline:

    def __init__(self):
        rospy.init_node('cedirnet_pipeline')
        init_tf()

        self.bridge = CvBridge()
        self.sample_name = None
        self.ok = False
        self.intrinsics = None   # (fx, fy, cx, cy) of the middle camera

        # publishers created once, before any callback can use them
        self.goal_pub = rospy.Publisher('/cedirnet/goal_pose', PoseStamped, queue_size=1, latch=True)
        self.finished_pub = rospy.Publisher('/cedirnet/finished', Bool, queue_size=1)

        rospy.Subscriber(TRIGGER_TOPIC, Int32, self.trigger_callback, queue_size=1)
        rospy.loginfo('CeDiRNet pipeline node ready.')

    # -------------------------
    # paths
    # -------------------------
    @property
    def sample_dir(self):
        return BASE_DIR / self.sample_name

    @property
    def obs_dir(self):
        return self.sample_dir / 'observation_start'

    @staticmethod
    def next_sample_name():
        nums = []
        for d in BASE_DIR.iterdir():
            m = re.fullmatch(r'sample_(\d{6})', d.name)
            if m:
                nums.append(int(m.group(1)))
        return f'sample_{(max(nums) + 1 if nums else 0):06d}'

    # -------------------------
    # capture
    # -------------------------
    def capture(self, n):
        """Fresh RGB + depth -> image_<n>.png, depth_<n>.tiff (float32 metres)."""
        rgb_msg = rospy.wait_for_message(RGB_TOPIC, Image, timeout=10)
        depth_msg = rospy.wait_for_message(DEPTH_TOPIC, Image, timeout=10)

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        cv2.imwrite(str(self.obs_dir / f'image_{n}.png'), rgb)
        cv2.imwrite(str(self.obs_dir / f'depth_{n}.tiff'), to_meters(depth_msg, self.bridge))
        rospy.loginfo(f'Saved image_{n}.png / depth_{n}.tiff')
        return depth_msg

    def save_json(self, name, data, indent=2):
        with open(self.obs_dir / name, 'w') as f:
            json.dump(data, f, indent=indent)

    def save_pose(self, filename, target, source):
        tf = getTfTransform(target, source, returnMatrix=False)
        if tf is None:
            raise RuntimeError(f'Could not obtain {target} -> {source} transform')
        T, Q = tf
        roll, pitch, yaw = tft.euler_from_quaternion(Q)
        self.save_json(filename, {
            'position_in_meters': {'x': float(T[0]), 'y': float(T[1]), 'z': float(T[2])},
            'rotation_euler_xyz_in_radians': {'roll': float(roll), 'pitch': float(pitch), 'yaw': float(yaw)},
        })

    def save_middle_extras(self, depth_msg):
        """Everything saveData.py used to write, taken at the middle head position."""
        joints = rospy.wait_for_message('/joint_states', JointState, timeout=10)
        info = rospy.wait_for_message(INFO_TOPIC, CameraInfo, timeout=10)

        self.save_json('arm_left_joints.json', {'values': list(joints.position[0:7])}, indent=4)
        self.save_json('arm_right_joints.json', {'values': list(joints.position[7:14])}, indent=4)

        self.save_pose('arm_left_pose_in_world.json', 'odom', 'arm_left_1_link')
        self.save_pose('arm_left_tcp_pose_in_world.json', 'odom', 'gripper_left_base_link')
        self.save_pose('arm_right_pose_in_world.json', 'odom', 'arm_right_1_link')
        self.save_pose('arm_right_tcp_pose_in_world.json', 'odom', 'gripper_right_base_link')
        self.save_pose('camera_pose_in_world.json', 'odom', 'rgbd_depth_optical_frame')

        # camera_intrinsics.json is written in make_panorama() (resolution changes)
        self.intrinsics = (info.K[0], info.K[4], info.K[2], info.K[5])

        self.save_json('requested_model.json', {'requested_model': REQUESTED_MODEL}, indent=4)

        raw_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        cv2.imwrite(str(self.obs_dir / 'depth_image.jpg'), raw_depth)

        ImagePIL.new('RGB', (854, 480), color=(0, 0, 0)).save(self.obs_dir / 'confidence_map.tiff')
        rospy.loginfo('Middle-position data saved')

    # -------------------------
    # panorama
    # -------------------------
    def make_panorama(self):
        imgs = [cv2.imread(str(self.obs_dir / f'image_{i}.png')) for i in (1, 2, 3)]
        deps = [cv2.imread(str(self.obs_dir / f'depth_{i}.tiff'), cv2.IMREAD_UNCHANGED) for i in (1, 2, 3)]
        if any(x is None for x in imgs + deps):
            raise RuntimeError('Missing image/depth files for panorama')

        h, w = imgs[1].shape[:2]
        if PAN_OFFSET_Y + h > 1024 or PAN_OFFSET_X + w > 1024:
            raise RuntimeError(f'Image {w}x{h} does not fit the 1024x1024 stitching canvas')

        rgb_pan, depth_pan = stitch_three(imgs[0], imgs[1], imgs[2], deps[0], deps[1], deps[2])

        pad = dict(top=PAD_TOP, bottom=PAD_BOTTOM, left=PAD_LEFT, right=PAD_RIGHT,
                   borderType=cv2.BORDER_CONSTANT)
        rgb_small = cv2.resize(rgb_pan, (PAN_SIZE, PAN_SIZE))
        # nearest neighbour for depth: interpolation would invent depths at object edges
        depth_small = cv2.resize(depth_pan, (PAN_SIZE, PAN_SIZE), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(self.obs_dir / 'image_left.png'),
                    cv2.copyMakeBorder(rgb_small, value=(0, 0, 0), **pad))
        cv2.imwrite(str(self.obs_dir / 'depth_map.tiff'),
                    cv2.copyMakeBorder(depth_small, value=0, **pad))

        fx, fy, cx, cy = self.intrinsics
        if ADJUST_INTRINSICS:
            s = PAN_SIZE / 1024.0
            fx, fy = fx * s, fy * s
            cx = s * (cx + PAN_OFFSET_X) + PAD_LEFT
            cy = s * (cy + PAN_OFFSET_Y) + PAD_TOP

        self.save_json('camera_intrinsics.json', {
            'image_resolution': {'width': FINAL_SIZE, 'height': FINAL_SIZE},
            'focal_lengths_in_pixels': {'fx': fx, 'fy': fy},
            'principal_point_in_pixels': {'cx': cx, 'cy': cy},
        }, indent=4)
        rospy.loginfo('Panorama saved as image_left.png / depth_map.tiff')

    # -------------------------
    # remote processing
    # -------------------------
    @staticmethod
    def run(cmd):
        if subprocess.run(cmd).returncode != 0:
            raise RuntimeError(f'Command failed: {" ".join(cmd)}')

    def remote(self):
        return f'{REMOTE_USER}@{REMOTE_HOST}'

    def copy_to_remote(self):
        # a leftover remote folder with the same name would make wait_for_result() return a stale result
        r = subprocess.run(['ssh', self.remote(), f'test ! -e {REMOTE_PATH}/{self.sample_name}'])
        if r.returncode != 0:
            raise RuntimeError(f'{self.sample_name} already exists on the remote machine')
        self.run(['scp', '-r', f'{self.sample_dir}/', f'{self.remote()}:{REMOTE_PATH}'])

    def wait_for_result(self):
        remote_json = f'{REMOTE_PATH}/{self.sample_name}/grasp_predicted/grasp_pose.json'
        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            r = subprocess.run(['ssh', self.remote(), f'test -f {remote_json}'])
            if r.returncode == 0:
                rospy.loginfo('Result found!')
                return
            if (rospy.Time.now() - t0).to_sec() > RESULT_TIMEOUT:
                raise RuntimeError('Timeout waiting for remote result')
            rospy.loginfo('Waiting for result...')
            rospy.sleep(1.0)

    def copy_result_back(self):
        self.run(['scp', '-r',
                  f'{self.remote()}:{REMOTE_PATH}/{self.sample_name}/grasp_predicted/',
                  f'{self.sample_dir}/'])

    def publish_best_grasp(self):
        with open(self.sample_dir / 'grasp_predicted' / 'grasp_pose.json') as f:
            data = json.load(f)
        best = max(data, key=lambda x: x['score'])
        pos = best['position_in_meters']
        rot = best['rotation_euler_xyz_in_radians']
        q = tft.quaternion_from_euler(rot['roll'], rot['pitch'], rot['yaw'])

        msg = PoseStamped()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pos['x'], pos['y'], pos['z']
        msg.pose.orientation.x, msg.pose.orientation.y = q[0], q[1]
        msg.pose.orientation.z, msg.pose.orientation.w = q[2], q[3]
        self.goal_pub.publish(msg)
        rospy.loginfo('Goal pose published')

    # -------------------------
    # trigger
    # -------------------------
    def trigger_callback(self, msg):
        n = msg.data
        if n not in (1, 2, 3):
            rospy.logwarn(f'Unexpected trigger value {n}')
            return
        rospy.loginfo(f'Trigger {n} received')

        try:
            if n == 1:
                self.ok = False
                self.sample_name = self.next_sample_name()
                self.obs_dir.mkdir(parents=True, exist_ok=True)
                rospy.loginfo(f'New sample: {self.sample_name}')
                self.capture(1)
                self.ok = True
            elif not self.ok:
                raise RuntimeError('Previous step failed (or trigger 1 missing) - skipping')
            elif n == 2:
                depth_msg = self.capture(2)
                self.save_middle_extras(depth_msg)
            else:
                self.capture(3)
                self.make_panorama()
                self.copy_to_remote()
                self.wait_for_result()
                self.copy_result_back()
                self.publish_best_grasp()
                rospy.loginfo('Done')
        except Exception as e:
            self.ok = False
            rospy.logerr(f'Pipeline error at step {n}: {e}')
        finally:
            self.finished_pub.publish(Bool(data=True))   # never leave the head loop hanging


if __name__ == '__main__':
    CedirnetPipeline()
    rospy.spin()
