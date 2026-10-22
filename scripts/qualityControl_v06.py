#!/usr/bin/env python3

import cv2
import json
import rospy
import tf2_ros
import subprocess
import numpy as np
import tf.transformations as tft
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as ScipyRot
from pathlib import Path
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Int32


# ----------------------------------------------------------------------
# Pose-based panorama (no feature matching needed)
# ----------------------------------------------------------------------

BLEND_POWER = 2      # >1 narrows the blending band between images (less ghosting)


def _gain_compensation(warped, weights, sigma_rel=0.03, sigma_g=0.5, min_overlap=500):
    '''
    Per-image, per-channel gains that equalise the overlap regions
    (Brown & Lowe style least squares with a weak prior pulling gains towards 1).
    The error is relative to the overlap brightness, so dark and bright cloth
    behave the same. Compensates auto-exposure / white-balance changes between shots.
    '''
    n = len(warped)
    valid = [wt > 1e-3 for wt in weights]
    stats = []
    for i in range(n):
        for j in range(i + 1, n):
            ov = valid[i] & valid[j]
            N = int(ov.sum())
            if N >= min_overlap:
                stats.append((i, j, N,
                              warped[i][ov].astype(np.float64).mean(axis=0),
                              warped[j][ov].astype(np.float64).mean(axis=0)))
    gains = np.ones((n, 3))
    if not stats:
        return gains

    for c in range(3):
        A, b = [], []
        prior_n = np.zeros(n)
        for i, j, N, mi, mj in stats:
            row = np.zeros(n)
            scale = max(0.5 * (mi[c] + mj[c]), 1.0)
            row[i], row[j] = mi[c] / scale, -mj[c] / scale
            A.append(row * np.sqrt(N) / sigma_rel)
            b.append(0.0)
            prior_n[i] += N
            prior_n[j] += N
        for k in range(n):
            row = np.zeros(n)
            row[k] = np.sqrt(prior_n[k]) / sigma_g
            A.append(row)
            b.append(np.sqrt(prior_n[k]) / sigma_g)
        g = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
        gains[:, c] = np.clip(g, 0.5, 2.0)
    return gains


def stitch_by_rotation(imgs, masks, Rs, K, scale=1.0, gain_comp=True, info=None):
    '''
    Stitch images taken by a (mostly) rotating camera onto a spherical canvas,
    using known camera orientations instead of feature matching.

    imgs  : list of undistorted BGR images (all same size)
    masks : list of uint8 masks (255 = valid pixel) for each image
    Rs    : list of 3x3 rotations, camera optical frame -> world
    K     : 3x3 intrinsics matching the (undistorted, cropped) images
    info  : optional dict, receives the exposure gains
    '''
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = imgs[0].shape[:2]
    Kinv = np.linalg.inv(K)

    # reference camera = the one looking in the "middle" direction
    axes = np.array([R[:, 2] for R in Rs])
    ref = int(np.argmax(axes @ axes.mean(axis=0)))
    R_ref = Rs[ref]

    # --- canvas extent: project image borders into the reference frame
    xs = np.linspace(0, w - 1, 40)
    ys = np.linspace(0, h - 1, 40)
    border = np.concatenate([
        np.stack([xs, np.zeros_like(xs)]), np.stack([xs, np.full_like(xs, h - 1)]),
        np.stack([np.zeros_like(ys), ys]), np.stack([np.full_like(ys, w - 1), ys])
    ], axis=1)
    border = np.vstack([border, np.ones(border.shape[1])])
    thetas, phis = [], []
    for R in Rs:
        d = R_ref.T @ R @ (Kinv @ border)
        thetas.append(np.arctan2(d[0], d[2]))
        phis.append(np.arctan2(d[1], np.hypot(d[0], d[2])))
    thetas, phis = np.concatenate(thetas), np.concatenate(phis)
    t_min, t_max, p_min, p_max = thetas.min(), thetas.max(), phis.min(), phis.max()

    s = fx * scale                                   # pixels per radian
    W = int(np.ceil((t_max - t_min) * s)) + 1
    H = int(np.ceil((p_max - p_min) * s)) + 1

    # --- ray for every canvas pixel (reference optical frame: x right, y down, z fwd)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    theta = t_min + u / s
    phi = p_min + v / s
    rays = np.stack([np.sin(theta) * np.cos(phi),
                     np.sin(phi),
                     np.cos(theta) * np.cos(phi)], axis=-1).astype(np.float64)

    warped, weights = [], []
    for img, mask, R in zip(imgs, masks, Rs):
        M = R.T @ R_ref                              # ref camera -> this camera
        d = rays @ M.T
        z = d[..., 2]
        front = z > 0.05
        zs = np.where(front, z, 1.0)
        map_x = np.clip(fx * d[..., 0] / zs + cx, -10, w + 10).astype(np.float32)
        map_y = np.clip(fy * d[..., 1] / zs + cy, -10, h + 10).astype(np.float32)

        # feathering weight = distance to the nearest invalid pixel
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        wt = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1].astype(np.float32)
        wt_w = cv2.remap(wt, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        wt_w[~front] = 0
        img_w = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped.append(img_w)
        weights.append(wt_w)

    gains = _gain_compensation(warped, weights) if gain_comp else np.ones((len(imgs), 3))
    if info is not None:
        info['gains_bgr'] = gains.tolist()

    acc = np.zeros((H, W, 3), np.float32)
    wsum = np.zeros((H, W), np.float32)
    for img_w, wt_w, g in zip(warped, weights, gains):
        wb = wt_w ** BLEND_POWER
        acc += img_w.astype(np.float32) * g.astype(np.float32)[None, None, :] * wb[..., None]
        wsum += wb

    valid = wsum > 1e-3
    pano = np.zeros((H, W, 3), np.uint8)
    pano[valid] = np.clip(acc[valid] / wsum[valid][:, None], 0, 255).astype(np.uint8)

    # crop to the bounding box of valid pixels
    ys_, xs_ = np.where(valid)
    pano = pano[ys_.min():ys_.max() + 1, xs_.min():xs_.max() + 1]
    return pano


# ----------------------------------------------------------------------
# Feature-based refinement of the TF poses
# ----------------------------------------------------------------------

def _sqdist(A, B):
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(d, 0)


def _detect_features(img, mask, n_features=3000):
    '''
    SIFT on a contrast-normalised copy. The brightening is adaptive: dark cloth
    is lifted strongly, bright cloth is left alone.
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    valid = mask > 0
    mean = float(np.clip(gray[valid].mean() if valid.any() else gray.mean(), 1.0, 254.0))
    gamma = float(np.clip(np.log(110.0 / 255.0) / np.log(mean / 255.0), 0.4, 1.0))
    lut = (255.0 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(cv2.LUT(gray, lut))

    m = cv2.erode(mask, np.ones((15, 15), np.uint8))
    m[:10], m[-10:], m[:, :10], m[:, -10:] = 0, 0, 0, 0

    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.01)
    kp, des = sift.detectAndCompute(gray, m)
    if des is None or len(kp) < 2:
        return np.zeros((0, 2), np.float32), None
    return np.float32([k.pt for k in kp]), des


def _guided_match(pts_i, des_i, pts_j, des_j, H, size, gate_px, ratio=0.8):
    '''
    Match keypoints of image i to image j, but only accept candidates that lie
    within gate_px of where the TF-predicted homography H says they should be.
    Ratio test + mutual check on top. Returns matched pixel coordinates.
    '''
    empty = np.zeros((0, 2), np.float32)
    if des_i is None or des_j is None or len(pts_i) < 2 or len(pts_j) < 2:
        return empty, empty

    w, h = size
    p = np.c_[pts_i, np.ones(len(pts_i))] @ H.T
    ok = p[:, 2] > 1e-3
    pred = p[:, :2] / np.where(ok, p[:, 2], 1.0)[:, None]
    ok &= ((pred[:, 0] > -gate_px) & (pred[:, 0] < w + gate_px) &
           (pred[:, 1] > -gate_px) & (pred[:, 1] < h + gate_px))
    idx = np.where(ok)[0]
    if len(idx) < 2:
        return empty, empty

    D = _sqdist(des_i[idx], des_j)
    D[~(_sqdist(pred[idx], pts_j) < gate_px ** 2)] = np.inf

    two = np.partition(D, 1, axis=1)[:, :2]
    best = D.argmin(axis=1)
    back = D.argmin(axis=0)
    keep = (np.isfinite(two[:, 0]) &
            (two[:, 0] < ratio ** 2 * two[:, 1]) &
            (back[best] == np.arange(len(idx))))
    return pts_i[idx[keep]], pts_j[best[keep]]


def _validate_pair(xi, xj, min_inliers=12, ransac_px=3.0, ambiguity_ratio=0.5):
    '''
    Decide whether the matches of one image pair can be trusted.

    A homography is fitted with RANSAC. Then a second one is fitted to the
    leftover matches. On periodic cloth (stripes, checks) wrong matches that
    are shifted by one pattern period form a second, comparably strong model;
    such a pair is ambiguous and is not used (the TF poses stay in charge).

    Returns (inlier_mask or None, status, n_inliers, n_second_model).
    '''
    if len(xi) < min_inliers:
        return None, 'too_few_matches', 0, 0
    H1, m1 = cv2.findHomography(xi, xj, cv2.RANSAC, ransac_px, maxIters=2000, confidence=0.995)
    if H1 is None or m1 is None:
        return None, 'no_model', 0, 0
    m1 = m1.ravel().astype(bool)
    n1 = int(m1.sum())
    if n1 < min_inliers:
        return None, 'too_few_inliers', n1, 0

    n2 = 0
    rest = ~m1
    if rest.sum() >= 8:
        H2, m2 = cv2.findHomography(xi[rest], xj[rest], cv2.RANSAC, ransac_px, maxIters=2000)
        if H2 is not None and m2 is not None:
            n2 = int(m2.sum())
    if n2 >= ambiguity_ratio * n1:
        return None, 'ambiguous', n1, n2
    return m1, 'ok', n1, n2


def refine_rotations(imgs, masks, Rs, K, max_corr_deg=3.0, min_matches=12,
                     max_matches_per_pair=300, sigma_px=2.0, sigma_rot=0.03,
                     ransac_px=3.0, ambiguity_ratio=0.5, outlier_px=6.0, log=print):
    '''
    Refine camera rotations (camera->world) that came from TF.

    Every image gets a small correction rotation (right-multiplied, rotation
    vector delta_k, limited to max_corr_deg). The correction minimises the
    reprojection error of feature matches between overlapping images, with a
    prior pulling every delta towards zero (= trust TF). Images / pairs
    without trustworthy matches therefore keep their TF pose.
    Returns (new_rotations, info_dict).
    '''
    n = len(imgs)
    h, w = imgs[0].shape[:2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Kinv = np.linalg.inv(K)
    rng = np.random.default_rng(0)

    max_corr = np.radians(max_corr_deg)
    gate_px = 1.5 * fx * np.tan(max_corr)          # matching gate follows the allowed correction
    info = {'applied': False, 'gate_px': float(gate_px), 'pairs': {}}

    feats = [_detect_features(im, m) for im, m in zip(imgs, masks)]

    def rays_of(pix):
        return (Kinv @ np.c_[pix, np.ones(len(pix))].T).T

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if Rs[i][:, 2] @ Rs[j][:, 2] < np.cos(np.radians(85)):
                continue                                   # looking in (nearly) opposite directions
            H = K @ Rs[j].T @ Rs[i] @ Kinv
            xi, xj = _guided_match(feats[i][0], feats[i][1], feats[j][0], feats[j][1],
                                   H, (w, h), gate_px)
            ok, status, n1, n2 = _validate_pair(xi, xj, min_matches, ransac_px, ambiguity_ratio)
            info['pairs'][f'{i}-{j}'] = {'matches': int(len(xi)), 'inliers': n1,
                                         'second_model': n2, 'status': status}
            if status != 'ok':
                continue
            xi, xj = xi[ok], xj[ok]
            if len(xi) > max_matches_per_pair:
                sel = rng.choice(len(xi), max_matches_per_pair, replace=False)
                xi, xj = xi[sel], xj[sel]
            pairs.append((i, j, rays_of(xi), xi, rays_of(xj), xj))

    summary = {k: f"{v['status']}({v['inliers']})" for k, v in info['pairs'].items()}
    log(f'refine: pairs {summary}')
    if not pairs:
        log('refine: no pair is usable, keeping TF poses')
        return list(Rs), info

    def rotmats(delta):
        return [Rs[k] @ ScipyRot.from_rotvec(delta[3 * k:3 * k + 3]).as_matrix() for k in range(n)]

    def project(rays, M):
        d = rays @ M.T
        z = np.maximum(d[:, 2], 1e-3)
        return np.stack([fx * d[:, 0] / z + cx, fy * d[:, 1] / z + cy], axis=1)

    def pair_res(Rn, p):
        i, j, ri, xi, rj, xj = p
        M = Rn[j].T @ Rn[i]
        return project(ri, M) - xj, project(rj, M.T) - xi      # forward, backward (pixels)

    def fun(delta, prs):
        Rn = rotmats(delta)
        res = []
        for p in prs:
            f, b = pair_res(Rn, p)
            res.append(f.ravel() / sigma_px)
            res.append(b.ravel() / sigma_px)
        res.append(delta / sigma_rot)                           # prior: stay close to TF
        return np.concatenate(res)

    def rms_px(delta, prs):
        Rn = rotmats(delta)
        e = np.concatenate([np.concatenate([np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)])
                            for a, b in (pair_res(Rn, p) for p in prs)])
        return float(np.sqrt(np.mean(e ** 2)))

    def solve(prs, x0):
        return least_squares(fun, x0, args=(prs,), loss='soft_l1', f_scale=1.0,
                             bounds=(-max_corr, max_corr), method='trf').x

    delta = solve(pairs, np.zeros(3 * n))

    # drop outlier matches using the first solution, then solve again
    Rn = rotmats(delta)
    clean = []
    for p in pairs:
        f, b = pair_res(Rn, p)
        ok = (np.linalg.norm(f, axis=1) < outlier_px) & (np.linalg.norm(b, axis=1) < outlier_px)
        if ok.sum() >= min_matches:
            clean.append((p[0], p[1], p[2][ok], p[3][ok], p[4][ok], p[5][ok]))
    if not clean:
        log('refine: all matches rejected as outliers, keeping TF poses')
        return list(Rs), info

    delta = solve(clean, delta)

    before = rms_px(np.zeros(3 * n), clean)
    after = rms_px(delta, clean)
    info.update({
        'rms_before_px': before,
        'rms_after_px': after,
        'inlier_matches': int(sum(len(p[3]) for p in clean)),
        'correction_deg': [float(np.degrees(np.linalg.norm(delta[3 * k:3 * k + 3]))) for k in range(n)],
    })
    log(f'refine: inlier reprojection RMS {before:.2f}px -> {after:.2f}px, '
        f'corrections [deg] {[round(c, 2) for c in info["correction_deg"]]}')

    if np.abs(delta).max() > 0.98 * max_corr:
        log(f'refine: a correction hit the {max_corr_deg:.1f} deg limit - TF poses are either far off '
            f'or the matches are aliased; keeping TF poses')
        return list(Rs), info
    if not after < before:
        log('refine: no improvement, keeping TF poses')
        return list(Rs), info

    info['applied'] = True
    return rotmats(delta), info


class qualityControl:

    NUM_IMAGES = 6

    # The camera image has black letterbox bars (top/bottom). They ruin
    # feature matching and blending, so they are cropped away. Tune if needed.
    CROP_TOP = 24
    CROP_BOTTOM = 28

    PANO_SCALE = 1.0          # 1.0 = native resolution at image centre
    WORLD_FRAME = 'odom'

    def __init__(self):

        self.bridge = CvBridge()

        self.remote_user = 'anamarija'
        self.remote_host = '178.172.42.49'
        self.remote_tmp = '/home/anamarija/SuperSimpleNet/tools/tmpServe'
        self.remote_serve = '/home/anamarija/SuperSimpleNet/tools/serve'
        self.base_dir = Path('/home/pal/docker_anamarija/dataForClothDefectService')

        self.result_timeout = 120.0
        self.folder_name = None
        self.sample_idx = None
        self.sample_dir = None
        self.upload_dir = None
        self.images = []
        self.rotations = []       # camera->world rotation for every image (or None)

        self.use_tf = rospy.get_param('~use_tf', True)
        self.refine = rospy.get_param('~refine', True)           # feature refinement of TF poses
        self.save_debug = rospy.get_param('~save_debug', True)   # TF-only panorama + refinement.json
        self.gain_comp = rospy.get_param('~gain_comp', True)     # exposure / white-balance matching
        self.max_corr_deg = rospy.get_param('~max_corr_deg', 3.0)  # max. pose correction per image
        self.K = None
        self.D = None
        self.valid_mask = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.saved_pub = rospy.Publisher('/quality_control/image_saved', Bool, queue_size=1)
        rospy.Subscriber('/quality_control/trigger', Int32, self.trigger_callback)

    # ------------------------------------------------------------------
    # triggers / capture
    # ------------------------------------------------------------------

    def trigger_callback(self, msg):

        idx = msg.data

        try:
            if idx == 1:
                self.start_sample()

            if self.folder_name is None:
                rospy.logwarn(f'Trigger {idx} received but no sample started (missing trigger 1)')
                return

            self.capture_image(idx)

            self.saved_pub.publish(True)

            if len(self.images) == self.NUM_IMAGES:
                rospy.loginfo('All images collected, running quality control...')
                self.process()

        except Exception as e:
            rospy.logerr(f'Quality control failed: {e}')

    def run_cmd(self, cmd):

        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            raise RuntimeError(f'Command failed: {cmd}')

    def start_sample(self):

        self.images = []
        self.rotations = []

        self.base_dir.mkdir(parents=True, exist_ok=True)

        existing = [int(p.name.split('_')[1])
                    for p in self.base_dir.glob('sample_[0-9]*') if p.is_dir()]
        self.sample_idx = max(existing, default=0) + 1

        self.folder_name = f'sample_{self.sample_idx:04d}'
        self.sample_dir = self.base_dir / self.folder_name
        self.upload_dir = self.sample_dir / self.folder_name
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        rospy.loginfo(f'Started {self.folder_name}')

    def load_intrinsics(self):

        if self.K is not None:
            return
        info = rospy.wait_for_message('/rgbd/rgb/camera_info', CameraInfo, timeout=5)
        self.K = np.array(info.K, dtype=np.float64).reshape(3, 3)
        self.D = np.array(info.D, dtype=np.float64)

        # valid-pixel mask of the undistorted image (undistortion leaves black corners)
        h, w = info.height, info.width
        m = cv2.undistort(np.full((h, w), 255, np.uint8), self.K, self.D)
        m = (m > 250).astype(np.uint8) * 255
        m = cv2.erode(m, np.ones((5, 5), np.uint8))
        self.valid_mask = m[self.CROP_TOP:h - self.CROP_BOTTOM]

    def get_camera_rotation(self, msg):
        '''Camera optical frame -> world rotation at the moment of capture.'''

        try:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.WORLD_FRAME, msg.header.frame_id, msg.header.stamp, rospy.Duration(0.5))
            except tf2_ros.TransformException:
                t = self.tf_buffer.lookup_transform(
                    self.WORLD_FRAME, msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
        except tf2_ros.TransformException as e:
            rospy.logwarn(f'TF lookup failed for image: {e}')
            return None

        q = t.transform.rotation
        return tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

    def capture_image(self, idx):

        rospy.loginfo(f'Capturing image {idx}/{self.NUM_IMAGES}...')
        rgb_msg = rospy.wait_for_message('/rgbd/rgb/image_raw', Image, timeout=10)
        cv_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        self.images.append(cv_img)
        self.rotations.append(self.get_camera_rotation(rgb_msg) if self.use_tf else None)
        cv2.imwrite(str(self.sample_dir / f'image_{self.sample_idx:04d}_{idx}.png'), cv_img)

    # ------------------------------------------------------------------
    # panorama
    # ------------------------------------------------------------------

    def preprocess(self, img):
        '''Undistort and crop the letterbox bars.'''
        h = img.shape[0]
        und = cv2.undistort(img, self.K, self.D)
        return und[self.CROP_TOP:h - self.CROP_BOTTOM]

    def make_panorama_pose(self):

        self.load_intrinsics()
        imgs = [self.preprocess(im) for im in self.images]
        masks = [self.valid_mask] * len(imgs)

        K = self.K.copy()
        K[1, 2] -= self.CROP_TOP          # principal point shifts after cropping

        Rs = self.rotations
        debug = {}

        if self.refine:
            if self.save_debug:
                tf_only = stitch_by_rotation(imgs, masks, Rs, K, self.PANO_SCALE, gain_comp=False)
                cv2.imwrite(str(self.sample_dir / f'panorama_{self.sample_idx:04d}_tf_only.png'), tf_only)
            try:
                Rs, debug = refine_rotations(imgs, masks, Rs, K, max_corr_deg=self.max_corr_deg,
                                             log=rospy.loginfo)
            except Exception as e:
                rospy.logwarn(f'Pose refinement failed ({e}), using TF poses only')
                Rs = self.rotations

        pano = stitch_by_rotation(imgs, masks, Rs, K, self.PANO_SCALE,
                                  gain_comp=self.gain_comp, info=debug)
        rospy.loginfo(f'Pose-based panorama created, size {pano.shape[1]}x{pano.shape[0]}, '
                      f'exposure gains (BGR) {np.round(debug.get("gains_bgr", []), 2).tolist()}')

        if self.save_debug:
            with open(self.sample_dir / 'refinement.json', 'w') as f:
                json.dump(debug, f, indent=4)
        return pano

    def make_panorama_features(self):
        '''Fallback: OpenCV feature-based stitcher. Checks that ALL images were used.'''

        h = self.images[0].shape[0]
        imgs = [im[self.CROP_TOP:h - self.CROP_BOTTOM] for im in self.images]

        for mode in (cv2.Stitcher_SCANS, cv2.Stitcher_PANORAMA):
            stitcher = cv2.Stitcher_create(mode)
            stitcher.setPanoConfidenceThresh(0.3)       # default 1.0 silently drops weak images
            status, pano = stitcher.stitch(imgs)
            if status != cv2.Stitcher_OK:
                rospy.logwarn(f'Stitching with mode {mode} failed, status {status}')
                continue
            used = len(stitcher.component())
            if used < len(imgs):
                rospy.logwarn(f'Stitcher (mode {mode}) used only {used}/{len(imgs)} images')
                continue
            rospy.loginfo(f'Feature panorama created (mode {mode}), size {pano.shape[1]}x{pano.shape[0]}')
            return pano
        raise RuntimeError('Feature-based panorama stitching failed')

    def make_panorama(self):

        if self.use_tf and all(R is not None for R in self.rotations):
            try:
                return self.make_panorama_pose()
            except Exception as e:
                rospy.logwarn(f'Pose-based stitching failed ({e}), falling back to features')
        else:
            rospy.logwarn('Missing TF rotations, using feature-based stitching')
        return self.make_panorama_features()

    def save_panorama(self, pano):

        cv2.imwrite(str(self.upload_dir / f'image_{self.sample_idx:04d}.png'), pano)
        rospy.loginfo(f'Saved panorama to {self.upload_dir}')

    # ------------------------------------------------------------------
    # remote inference
    # ------------------------------------------------------------------

    def copy_folder_to_remote(self):

        upload_name = f'upload_{self.folder_name}'
        target = f'{self.remote_user}@{self.remote_host}'
        staging = f'{self.remote_tmp}/{upload_name}'

        self.run_cmd(f"ssh {target} 'mkdir -p {staging}'")
        self.run_cmd(f'scp -r {self.upload_dir} {target}:{staging}/observation_start')
        self.run_cmd(f"ssh {target} 'mv {staging} {self.remote_tmp}/{self.folder_name}'")

    def wait_for_result(self):

        remote_json = f'{self.remote_serve}/{self.folder_name}/defect_predicted/result.json'
        target = f'{self.remote_user}@{self.remote_host}'
        start = rospy.Time.now()

        while not rospy.is_shutdown():
            result = subprocess.run(f"ssh {target} 'test -f {remote_json}'", shell=True)
            if result.returncode == 0:
                rospy.loginfo('Result found!')
                return
            if (rospy.Time.now() - start).to_sec() > self.result_timeout:
                raise RuntimeError(f'Timeout waiting for {remote_json}')
            rospy.loginfo('Waiting for result...')
            rospy.sleep(1)

    def copy_result_back(self):

        target = f'{self.remote_user}@{self.remote_host}'
        self.run_cmd(
            f'scp -r {target}:{self.remote_serve}/{self.folder_name}/defect_predicted/ '
            f'{self.sample_dir}/'
        )

    def read_result(self):

        result_file = self.sample_dir / 'defect_predicted' / 'result.json'
        with open(result_file, 'r') as f:
            result = json.load(f)
        rospy.loginfo(f'Result: {result}')
        return result

    def process(self):

        pano = self.make_panorama()
        self.save_panorama(pano)
        self.copy_folder_to_remote()
        self.wait_for_result()
        self.copy_result_back()
        self.read_result()


if __name__ == '__main__':

    rospy.init_node('qualityControl_node')

    QC = qualityControl()
    rospy.loginfo('qualityControl node ready, waiting for trigger on /quality_control/trigger')
    rospy.spin()
