#!/usr/bin/env python3

import cv2
import json
import rospy
import resource
import tf2_ros
import subprocess
import numpy as np
import multiprocessing as mp
import tf.transformations as tft
from pathlib import Path
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Int32
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as ScipyRot


# ----------------------------------------------------------------------
# Geometry: the cloth is treated as a plane, camera poses = full TF poses
# ----------------------------------------------------------------------

class _PlaneModel:
    '''Plane n.x = d with a small-angle parametrisation (alpha, beta, delta_d) around (n0, d0).'''

    def __init__(self, n0, d0):
        self.n0 = np.asarray(n0, float) / np.linalg.norm(n0)
        self.d0 = float(d0)
        a = np.cross(self.n0, [0.0, 0.0, 1.0])
        if np.linalg.norm(a) < 1e-6:
            a = np.cross(self.n0, [1.0, 0.0, 0.0])
        self.a1 = a / np.linalg.norm(a)
        self.a2 = np.cross(self.n0, self.a1)

    def params(self, th):
        n = self.n0 + th[0] * self.a1 + th[1] * self.a2
        return n / np.linalg.norm(n), self.d0 + th[2]


def _plane_intersect(rays_w, c, n, d):
    '''Intersect rays c + lam * rays_w with the plane n.x = d. Returns (points, lam).'''
    denom = rays_w @ n
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    lam = (d - n @ c) / denom
    return c + lam[:, None] * rays_w, lam


def _project_cam(X, R, c, K):
    '''Project world points into a camera with rotation R (cam->world) and centre c.'''
    Xc = (X - c) @ R                                   # = R^T (X - c)
    z = Xc[:, 2]
    zz = np.maximum(z, 1e-3)
    pix = np.stack([K[0, 0] * Xc[:, 0] / zz + K[0, 2], K[1, 1] * Xc[:, 1] / zz + K[1, 2]], axis=1)
    return pix, z > 0.05


def _transfer_rays(rays_cam, Ri, ci, Rj, cj, n, d, K):
    '''Pixels of image i (given as camera rays) -> pixels in image j, via the cloth plane.'''
    X, lam = _plane_intersect(rays_cam @ Ri.T, ci, n, d)
    pix, ok = _project_cam(X, Rj, cj, K)
    return pix, ok & (lam > 0.05)


# ----------------------------------------------------------------------
# Compositing on the cloth plane
# ----------------------------------------------------------------------

BLEND_POWER = 6      # high power = near winner-takes-all blending: seams instead of ghosts


def _edge_magnitude(gray):
    gb = cv2.GaussianBlur(gray, (0, 0), 1.2)
    return cv2.magnitude(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1))


def _photometric_compensation(warped, weights, min_overlap=3000, min_edge_ncc=0.3):
    '''
    Per-image, per-channel affine tone correction  I' = a * I + b  that makes overlapping
    images agree. Needed because auto exposure / auto contrast and glare from bright
    backgrounds change BOTH the gain and the black level between the shots (a gain-only
    model cannot remove a lifted black level: dark cloth turns grey in some views).

    - for every overlapping pair the tone mapping is fitted through matching quantiles of
      the overlap pixels (the overlap shows the same scene, so the distributions must agree)
    - a pair is only used if its edges correlate (>= min_edge_ncc). Edge correlation does not
      care about tone shifts or saturated whites, unlike plain intensity correlation
    - the pair mappings are chained along a spanning tree that prefers large, well aligned
      overlaps (no inconsistent loops that could shrink the contrast)
    - the root is the image with the largest tonal range (least haze)
    Returns an array (n, 3, 2) of (a, b) per image and channel.
    '''
    n = len(warped)
    comp = np.zeros((n, 3, 2))
    comp[:, :, 0] = 1.0
    valid = [wt > 1e-3 for wt in weights]
    gray = [cv2.cvtColor(wi, cv2.COLOR_BGR2GRAY) for wi in warped]
    mags = [_edge_magnitude(gi) for gi in gray]

    ranges = []
    for gi, vi in zip(gray, valid):
        px = gi[vi][::11]
        ranges.append(np.percentile(px, 99) - np.percentile(px, 1) if px.size > 100 else -1.0)
    root = int(np.argmax(ranges))

    qs = np.linspace(2, 98, 25)
    pair = {}
    for i in range(n):
        for j in range(i + 1, n):
            ov = cv2.erode((valid[i] & valid[j]).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            N = int(ov.sum())
            if N < min_overlap:
                continue
            ea, eb = mags[i][ov][::5], mags[j][ov][::5]
            if ea.std() < 1e-3 or eb.std() < 1e-3:
                continue
            q = float(np.corrcoef(ea, eb)[0, 1])
            if not q >= min_edge_ncc:
                continue
            maps = []                                   # i ~ a * j + b, per channel
            for c in range(3):
                qi = np.percentile(warped[i][..., c][ov][::7], qs)
                qj = np.percentile(warped[j][..., c][ov][::7], qs)
                if np.ptp(qj) < 1.0:
                    a = 1.0
                else:
                    a = float(np.clip(np.linalg.lstsq(np.stack([qj, np.ones_like(qj)], 1), qi, rcond=None)[0][0],
                                      0.5, 2.0))
                maps.append((a, float(np.median(qi - a * qj))))
            pair[(i, j)] = (N * q, maps)

    done = {root}
    while True:
        best = None
        for (i, j), (wgt, maps) in pair.items():
            if (i in done) != (j in done) and (best is None or wgt > best[0]):
                best = (wgt, i, j, maps)
        if best is None:
            break
        _, i, j, maps = best
        for c in range(3):
            a, b = maps[c]
            if i in done:             # j -> i scale:  I_i ~ a * I_j + b
                p, k = i, j
            else:                     # i -> j scale: invert the mapping
                p, k = j, i
                a, b = 1.0 / a, -b / a
            ap, bp = comp[p, c]
            comp[k, c] = (ap * a, ap * b + bp)
        done.add(j if i in done else i)
    comp[:, :, 0] = np.clip(comp[:, :, 0], 0.25, 4.0)
    return comp


def _alignment_score(warped, weights, comp, min_overlap=3000):
    '''
    How well do the warped images agree where they overlap? Weighted mean normalised
    correlation of the gradient magnitudes (1 = perfectly aligned, ~0 = misaligned / ghosted).
    Gradients make it independent of exposure; only edges and texture count.
    '''
    n = len(warped)
    mags, valid = [], []
    for img_w, wt, cp in zip(warped, weights, comp):
        gray = cv2.cvtColor(np.clip(img_w.astype(np.float32) * cp[None, None, :, 0] + cp[None, None, :, 1],
                                    0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        mags.append(_edge_magnitude(gray))
        valid.append(cv2.erode((wt > 1e-3).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    num, den = 0.0, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ov = valid[i] & valid[j]
            N = int(ov.sum())
            if N < min_overlap:
                continue
            a, b = mags[i][ov][::5], mags[j][ov][::5]
            if a.std() < 1e-3 or b.std() < 1e-3:
                continue
            num += np.sqrt(N) * float(np.corrcoef(a, b)[0, 1])
            den += np.sqrt(N)
    return float(num / den) if den > 0 else float('nan')


def stitch_on_plane(imgs, masks, Rs, cs, n, d, K, gain_comp=True, max_pixels=6e6, info=None):
    '''
    Warp all images onto the cloth plane (n.x = d) and blend them.

    The result is an orthographic, parallax-free view of the cloth (the background
    is smeared, but it is not what we inspect). Orientation follows the most frontal
    camera image, so the panorama looks like the camera images (not rotated).

    imgs/masks : undistorted images and valid-pixel masks
    Rs, cs     : camera rotations (cam->world) and centres in the world frame
    K          : intrinsics matching the images
    '''
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = imgs[0].shape[:2]
    Kinv = np.linalg.inv(K)

    # reference camera: the one looking most frontally at the plane (optical axis = -n)
    zs = np.array([R[:, 2] for R in Rs])
    ref = int(np.argmax(-(zs @ n)))
    X0, lam0 = _plane_intersect(Rs[ref][:, 2][None, :], cs[ref], n, d)
    p0, dist = X0[0], float(abs(lam0[0]))

    # canvas axes = camera x / y axes of the reference image, projected onto the plane
    ex = Rs[ref][:, 0] - (Rs[ref][:, 0] @ n) * n
    ex /= np.linalg.norm(ex)
    ey = Rs[ref][:, 1] - (Rs[ref][:, 1] @ n) * n - (Rs[ref][:, 1] @ ex) * ex
    ey /= np.linalg.norm(ey)

    # canvas extent: where the image borders hit the plane
    xs = np.linspace(0, w - 1, 40)
    ys = np.linspace(0, h - 1, 40)
    border = np.concatenate([
        np.stack([xs, np.zeros_like(xs)]), np.stack([xs, np.full_like(xs, h - 1)]),
        np.stack([np.zeros_like(ys), ys]), np.stack([np.full_like(ys, w - 1), ys])
    ], axis=1)
    rays0 = (Kinv @ np.vstack([border, np.ones(border.shape[1])])).T
    us, vs = [], []
    for R, c in zip(Rs, cs):
        X, lam = _plane_intersect(rays0 @ R.T, c, n, d)
        u, v = (X - p0) @ ex, (X - p0) @ ey
        ok = (lam > 0.05) & (np.abs(u) < 2.5) & (np.abs(v) < 2.5)
        us.append(u[ok])
        vs.append(v[ok])
    us, vs = np.concatenate(us), np.concatenate(vs)
    if len(us) < 10:
        raise RuntimeError('image borders do not hit the cloth plane')
    u_min, u_max, v_min, v_max = us.min(), us.max(), vs.min(), vs.max()

    s = fx / dist                                      # pixels per metre = native resolution
    W = int(np.ceil((u_max - u_min) * s)) + 1
    H = int(np.ceil((v_max - v_min) * s)) + 1
    if W * H > max_pixels:
        s *= np.sqrt(max_pixels / (W * H))
        W = int(np.ceil((u_max - u_min) * s)) + 1
        H = int(np.ceil((v_max - v_min) * s)) + 1

    uu = u_min + np.arange(W) / s
    vv = v_min + np.arange(H) / s
    X = (p0[None, None, :] + uu[None, :, None] * ex[None, None, :]
         + vv[:, None, None] * ey[None, None, :]).reshape(-1, 3)

    warped, weights = [], []
    for img, mask, R, c in zip(imgs, masks, Rs, cs):
        pix, front = _project_cam(X, R, c, K)
        map_x = np.clip(pix[:, 0], -10, w + 10).reshape(H, W).astype(np.float32)
        map_y = np.clip(pix[:, 1], -10, h + 10).reshape(H, W).astype(np.float32)

        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        wt = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1].astype(np.float32)
        wt_w = cv2.remap(wt, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        wt_w[~front.reshape(H, W)] = 0
        img_w = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped.append(img_w)
        weights.append(wt_w)

    if gain_comp:
        comp = _photometric_compensation(warped, weights)
    else:
        comp = np.zeros((len(imgs), 3, 2))
        comp[:, :, 0] = 1.0
    if info is not None:
        info['photometric_ab_bgr'] = comp.tolist()
        info['pano_pixels_per_metre'] = float(s)
        info['alignment_score'] = _alignment_score(warped, weights, comp)

    acc = np.zeros((H, W, 3), np.float32)
    wsum = np.zeros((H, W), np.float32)
    for img_w, wt_w, cp in zip(warped, weights, comp):
        wb = (wt_w / max(float(wt_w.max()), 1e-6)) ** BLEND_POWER
        corrected = np.clip(img_w.astype(np.float32) * cp[None, None, :, 0].astype(np.float32)
                            + cp[None, None, :, 1].astype(np.float32), 0, 255)
        acc += corrected * wb[..., None]
        wsum += wb

    valid = wsum > 1e-9
    pano = np.zeros((H, W, 3), np.uint8)
    pano[valid] = np.clip(acc[valid] / wsum[valid][:, None], 0, 255).astype(np.uint8)

    ys_, xs_ = np.where(valid)
    return pano[ys_.min():ys_.max() + 1, xs_.min():xs_.max() + 1]


# ----------------------------------------------------------------------
# Feature-based refinement of the TF poses and of the cloth plane
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


def _guided_match(pts_i, des_i, pred, ok, pts_j, des_j, size, gate_px, ratio=0.8):
    '''
    Match keypoints of image i to image j. Lowe's ratio test and the mutual check
    are done against ALL keypoints of j (so repeated texture cannot fool them);
    the accepted match must additionally lie within gate_px of the predicted
    position `pred` (from the current poses + plane). Returns matched pixels and their predictions.
    '''
    empty = np.zeros((0, 2), np.float32)
    if des_i is None or des_j is None or len(pts_i) < 2 or len(pts_j) < 2:
        return empty, empty, empty

    w, h = size
    ok = (ok & (pred[:, 0] > -gate_px) & (pred[:, 0] < w + gate_px) &
          (pred[:, 1] > -gate_px) & (pred[:, 1] < h + gate_px))
    idx = np.where(ok)[0]
    if len(idx) < 2:
        return empty, empty, empty

    D = _sqdist(des_i[idx], des_j)
    two = np.partition(D, 1, axis=1)[:, :2]
    best = D.argmin(axis=1)
    back = D.argmin(axis=0)
    keep = (two[:, 0] < ratio ** 2 * two[:, 1]) & (back[best] == np.arange(len(idx)))
    keep &= ((pred[idx] - pts_j[best]) ** 2).sum(axis=1) < gate_px ** 2
    return pts_i[idx[keep]], pts_j[best[keep]], pred[idx[keep]]


def _validate_pair(xi, xj, pred, min_inliers=12, ransac_px=3.0):
    '''
    Pick the matches that belong to the CLOTH.

    Matches of one image pair come from several surfaces (cloth close to the camera, the
    room far behind it), each forming its own homography. RANSAC finds these groups one by
    one; the cloth is the group that lies closest to the gripper-plane prediction `pred`
    (among groups with enough support), because anything far from the cloth plane shifts
    by tens of pixels when the head moves.

    Returns (inlier mask or None, status, n_inliers, n_groups).
    '''
    if len(xi) < min_inliers:
        return None, 'too_few_matches', 0, 0
    remaining = np.ones(len(xi), bool)
    groups = []
    for _ in range(3):
        idx = np.where(remaining)[0]
        if len(idx) < min_inliers:
            break
        H1, m1 = cv2.findHomography(xi[idx], xj[idx], cv2.RANSAC, ransac_px,
                                    maxIters=2000, confidence=0.995)
        if H1 is None or m1 is None:
            break
        sel = idx[m1.ravel().astype(bool)]
        if len(sel) < min_inliers:
            break
        groups.append(sel)
        remaining[sel] = False
    if not groups:
        return None, 'too_few_inliers', 0, 0

    biggest = max(len(g) for g in groups)
    cand = [g for g in groups if len(g) >= max(min_inliers, 0.2 * biggest)]
    disp = [float(np.median(np.linalg.norm(pred[g] - xj[g], axis=1))) for g in cand]
    best = cand[int(np.argmin(disp))]
    mask = np.zeros(len(xi), bool)
    mask[best] = True
    return mask, 'ok', int(len(best)), len(groups)


def refine_poses(imgs, masks, Rs, cs, n0, d0, K, max_corr_deg=6.0, search_px=40.0, narrow_px=20.0, min_matches=12,
                 max_matches_per_pair=300, sigma_px=2.0, sigma_rot=0.05, sigma_ang=0.25,
                 sigma_d=0.20, ransac_px=3.0, outlier_px=6.0, log=print):
    '''
    Refine, from feature matches,
      - a small rotation correction for every camera (<= max_corr_deg), and
      - the cloth plane (two tilt angles and the offset),
    starting from the TF poses (translation is trusted) and the gripper-based plane.
    Priors pull everything towards the TF values, so images / pairs without
    trustworthy matches keep their TF pose.

    Two passes: search window `search_px` first, then `narrow_px` around the first solution.
    Returns (Rs_new, n_new, d_new, info).
    '''
    n = len(imgs)
    h, w = imgs[0].shape[:2]
    fx = K[0, 0]
    Kinv = np.linalg.inv(K)
    rng = np.random.default_rng(0)

    max_corr = np.radians(max_corr_deg)
    # The search window is deliberately tight: points much further away than the cloth
    # (the room behind it) shift by 40-70 px relative to the cloth-plane prediction when the
    # head moves, so a wide window lets the plentiful background features hijack the fit.
    gate_wide = float(search_px)
    gate_narrow = float(narrow_px)
    plane = _PlaneModel(n0, d0)
    info = {'applied': False, 'gate_wide_px': float(gate_wide), 'pairs_wide': {}, 'pairs_narrow': {}}
    unchanged = (list(Rs), np.asarray(n0, float), float(d0), info)

    feats = [_detect_features(im, m) for im, m in zip(imgs, masks)]
    rays_kp = [(Kinv @ np.c_[f[0], np.ones(len(f[0]))].T).T for f in feats]

    # the gripper frame is not on the cloth (the clip and the cloth are in front of it), so the plane
    # offset gets a generous range; the rotation corrections stay small
    lo = np.r_[np.full(3 * n, -max_corr), [-0.5, -0.5, -0.5]]
    hi = -lo

    def poses(x):
        Rn = [Rs[k] @ ScipyRot.from_rotvec(x[3 * k:3 * k + 3]).as_matrix() for k in range(n)]
        nn, dd = plane.params(x[3 * n:])
        return Rn, nn, dd

    def build_pairs(x, gate):
        Rn, nn, dd = poses(x)
        prs, status = [], {}
        for i in range(n):
            for j in range(i + 1, n):
                if Rn[i][:, 2] @ Rn[j][:, 2] < np.cos(np.radians(85)):
                    continue
                pred, ok = _transfer_rays(rays_kp[i], Rn[i], cs[i], Rn[j], cs[j], nn, dd, K)
                xi, xj, pj = _guided_match(feats[i][0], feats[i][1], pred, ok,
                                           feats[j][0], feats[j][1], (w, h), gate)
                mask, st, n1, ng = _validate_pair(xi, xj, pj, min_matches, ransac_px)
                status[f'{i}-{j}'] = {'matches': int(len(xi)), 'inliers': n1, 'groups': ng, 'status': st}
                if st != 'ok':
                    continue
                xi, xj = xi[mask], xj[mask]
                if len(xi) > max_matches_per_pair:
                    sel = rng.choice(len(xi), max_matches_per_pair, replace=False)
                    xi, xj = xi[sel], xj[sel]
                ri = (Kinv @ np.c_[xi, np.ones(len(xi))].T).T
                rj = (Kinv @ np.c_[xj, np.ones(len(xj))].T).T
                prs.append((i, j, ri, xi, rj, xj))
        return prs, status

    def pair_res(Rn, nn, dd, p):
        i, j, ri, xi, rj, xj = p
        pij, ok1 = _transfer_rays(ri, Rn[i], cs[i], Rn[j], cs[j], nn, dd, K)
        pji, ok2 = _transfer_rays(rj, Rn[j], cs[j], Rn[i], cs[i], nn, dd, K)
        f = np.where(ok1[:, None], pij - xj, 30.0)
        b = np.where(ok2[:, None], pji - xi, 30.0)
        return f, b

    def fun(x, prs):
        Rn, nn, dd = poses(x)
        res = []
        for p in prs:
            f, b = pair_res(Rn, nn, dd, p)
            res.append(f.ravel() / sigma_px)
            res.append(b.ravel() / sigma_px)
        res.append(x[:3 * n] / sigma_rot)
        res.append(np.array([x[3 * n] / sigma_ang, x[3 * n + 1] / sigma_ang, x[3 * n + 2] / sigma_d]))
        return np.concatenate(res)

    def rms_px(x, prs):
        Rn, nn, dd = poses(x)
        e = np.concatenate([np.concatenate([np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)])
                            for a, b in (pair_res(Rn, nn, dd, p) for p in prs)])
        return float(np.sqrt(np.mean(e ** 2)))

    def solve(prs, x0):
        return least_squares(fun, x0, args=(prs,), loss='soft_l1', f_scale=1.0,
                             bounds=(lo, hi), method='trf').x

    def clean(prs, x):
        Rn, nn, dd = poses(x)
        out = []
        for p in prs:
            f, b = pair_res(Rn, nn, dd, p)
            ok = (np.linalg.norm(f, axis=1) < outlier_px) & (np.linalg.norm(b, axis=1) < outlier_px)
            if ok.sum() >= min_matches:
                out.append((p[0], p[1], p[2][ok], p[3][ok], p[4][ok], p[5][ok]))
        return out

    def short(status):
        return {k: f"{v['status']}({v['inliers']}/{v['groups']}g)" for k, v in status.items()}

    # ---- pass 1: wide window around the TF poses
    x = np.zeros(3 * n + 3)
    prs, status = build_pairs(x, gate_wide)
    info['pairs_wide'] = status
    log(f'refine (search {gate_wide:.0f}px): pairs {short(status)}')
    if not prs:
        log('refine: no pair is usable, keeping TF poses')
        return unchanged
    x = solve(prs, x)
    prs = clean(prs, x)
    if not prs:
        log('refine: all matches rejected as outliers, keeping TF poses')
        return unchanged
    x = solve(prs, x)

    # ---- pass 2: narrow window around the pass-1 solution (finds more, cleaner matches)
    prs2, status2 = build_pairs(x, gate_narrow)
    info['pairs_narrow'] = status2
    if prs2:
        x2 = solve(prs2, x)
        prs2 = clean(prs2, x2)
        if prs2:
            x, prs = solve(prs2, x2), prs2
    log(f'refine (narrow {gate_narrow:.0f}px): pairs {short(status2)}')

    before, after = rms_px(np.zeros_like(x), prs), rms_px(x, prs)
    nn, dd = plane.params(x[3 * n:])
    info.update({
        'rms_before_px': before, 'rms_after_px': after,
        'inlier_matches': int(sum(len(p[3]) for p in prs)),
        'rotation_correction_deg': [float(np.degrees(np.linalg.norm(x[3 * k:3 * k + 3]))) for k in range(n)],
        'plane_tilt_deg': [float(np.degrees(x[3 * n])), float(np.degrees(x[3 * n + 1]))],
        'plane_offset_cm': float(100 * x[3 * n + 2]),
    })
    log(f'refine: inlier reprojection RMS {before:.2f}px -> {after:.2f}px, '
        f'rotation corrections [deg] {[round(c, 2) for c in info["rotation_correction_deg"]]}, '
        f'plane tilt [deg] {[round(t, 1) for t in info["plane_tilt_deg"]]}, offset {info["plane_offset_cm"]:.1f} cm')

    at_limit = (np.abs(x[:3 * n]) > 0.98 * max_corr).any() or (np.abs(x[3 * n:]) > 0.98 * hi[3 * n:]).any()
    if at_limit:
        log('refine: a parameter hit its limit - the matches are not consistent with the TF geometry; keeping TF poses')
        return unchanged
    if not after < before:
        log('refine: no improvement, keeping TF poses')
        return unchanged

    Rn, nn, dd = poses(x)
    info['applied'] = True
    return Rn, nn, dd, info


# ----------------------------------------------------------------------
# OpenCV feature stitcher, run in a child process (it can silently use
# only some of the images, or allocate gigabytes for a garbage canvas)
# ----------------------------------------------------------------------

def _stitch_worker(imgs, mode, thr, conn):
    try:
        resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
        st = cv2.Stitcher_create(mode)
        st.setPanoConfidenceThresh(thr)
        status, pano = st.stitch(imgs)
        used = len(st.component()) if status == cv2.Stitcher_OK else 0
        conn.send((int(status), int(used), pano))
    except BaseException:
        conn.send((-1, 0, None))
    finally:
        conn.close()


def _isolated_stitch(imgs, mode, thr, timeout=90.0):
    ctx = mp.get_context('fork')
    rd, wr = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_stitch_worker, args=(imgs, mode, thr, wr), daemon=True)
    p.start()
    wr.close()
    res = None
    try:
        if rd.poll(timeout):
            res = rd.recv()
    except EOFError:
        res = None
    if p.is_alive():
        p.kill()
    p.join(1.0)
    return res


def feature_stitch(imgs, log=print, max_pixels=3.0e6, max_aspect=4.0):
    '''Feature-based panorama. Only a COMPLETE (all images used), plausibly sized result is accepted.'''
    for mode, thr in ((cv2.Stitcher_PANORAMA, 1.0), (cv2.Stitcher_PANORAMA, 0.5), (cv2.Stitcher_SCANS, 0.5)):
        res = _isolated_stitch(imgs, mode, thr)
        if res is None:
            log(f'feature stitcher (mode {mode}, thr {thr}) crashed or timed out')
            continue
        status, used, pano = res
        if status != cv2.Stitcher_OK or pano is None:
            log(f'feature stitcher (mode {mode}, thr {thr}) failed, status {status}')
            continue
        if used < len(imgs):
            log(f'feature stitcher (mode {mode}, thr {thr}) used only {used}/{len(imgs)} images')
            continue
        hh, ww = pano.shape[:2]
        if hh * ww > max_pixels or max(hh, ww) / max(min(hh, ww), 1) > max_aspect:
            log(f'feature stitcher (mode {mode}, thr {thr}) gave an implausible canvas {ww}x{hh}')
            continue
        log(f'feature panorama created (mode {mode}, thr {thr}), size {ww}x{hh}')
        return pano
    return None


class qualityControl:

    NUM_IMAGES = 6

    # The camera image has black letterbox bars (top/bottom). They ruin
    # feature matching and blending, so they are cropped away. Tune if needed.
    CROP_TOP = 24
    CROP_BOTTOM = 28

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
        self.rotations = []       # camera->world rotation of every image (or None)
        self.centers = []         # camera centre in the world frame (or None)

        self.use_tf = rospy.get_param('~use_tf', True)
        self.refine = rospy.get_param('~refine', True)             # feature refinement of TF poses + plane
        self.save_debug = rospy.get_param('~save_debug', True)     # TF-only panorama + debug json
        self.gain_comp = rospy.get_param('~gain_comp', True)       # exposure / white-balance matching
        self.max_corr_deg = rospy.get_param('~max_corr_deg', 6.0)  # max. rotation correction per image
        self.search_px = rospy.get_param('~search_px', 40.0)       # match search window around TF prediction
        self.narrow_px = rospy.get_param('~narrow_px', 20.0)
        self.plane_distance = rospy.get_param('~plane_distance', 0.6)   # only used if gripper TF is unavailable
        self.try_rotation_only = rospy.get_param('~try_rotation_only', True)  # try the model without TF translation
        self.try_translation = rospy.get_param('~try_translation', True)      # try the full TF pose + cloth plane model
        self.left_gripper_frame = rospy.get_param('~left_gripper_frame', 'gripper_left_base_link')
        self.right_gripper_frame = rospy.get_param('~right_gripper_frame', 'gripper_right_base_link')

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
        self.centers = []

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

    def lookup(self, frame, stamp=None):
        '''Pose of `frame` in the world frame -> (rotation 3x3, translation 3) or None.'''
        try:
            try:
                if stamp is None:
                    raise tf2_ros.TransformException('latest')
                t = self.tf_buffer.lookup_transform(self.WORLD_FRAME, frame, stamp, rospy.Duration(0.5))
            except tf2_ros.TransformException:
                t = self.tf_buffer.lookup_transform(self.WORLD_FRAME, frame, rospy.Time(0), rospy.Duration(1.0))
        except tf2_ros.TransformException as e:
            rospy.logwarn(f'TF lookup of {frame} failed: {e}')
            return None
        q, p = t.transform.rotation, t.transform.translation
        return tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3], np.array([p.x, p.y, p.z])

    def capture_image(self, idx):

        rospy.loginfo(f'Capturing image {idx}/{self.NUM_IMAGES}...')
        rgb_msg = rospy.wait_for_message('/rgbd/rgb/image_raw', Image, timeout=10)
        cv_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        self.images.append(cv_img)
        pose = self.lookup(rgb_msg.header.frame_id, rgb_msg.header.stamp) if self.use_tf else None
        self.rotations.append(None if pose is None else pose[0])
        self.centers.append(None if pose is None else pose[1])
        cv2.imwrite(str(self.sample_dir / f'image_{self.sample_idx:04d}_{idx}.png'), cv_img)

    # ------------------------------------------------------------------
    # panorama
    # ------------------------------------------------------------------

    def preprocess(self, img):
        '''Undistort and crop the letterbox bars.'''
        h = img.shape[0]
        und = cv2.undistort(img, self.K, self.D)
        return und[self.CROP_TOP:h - self.CROP_BOTTOM]

    def estimate_plane(self):
        '''
        Initial cloth plane n.x = d (n points towards the cameras). The cloth hangs from the
        two grippers, so it is the vertical plane through the line gripper-to-gripper.
        '''
        cs = np.array(self.centers)
        c_mean = cs.mean(axis=0)
        source = 'grippers'
        try:
            gl = self.lookup(self.left_gripper_frame)
            gr = self.lookup(self.right_gripper_frame)
            if gl is None or gr is None:
                raise RuntimeError('gripper TF unavailable')
            gl, gr = gl[1], gr[1]
            line = gr - gl
            n = np.cross(line, [0.0, 0.0, 1.0])
            if np.linalg.norm(line) < 0.1 or np.linalg.norm(n) < 1e-3:
                raise RuntimeError('degenerate gripper line')
            n /= np.linalg.norm(n)
            mid = 0.5 * (gl + gr)
        except Exception as e:
            rospy.logwarn(f'Plane from grippers failed ({e}); assuming a vertical plane '
                          f'{self.plane_distance} m in front of the head')
            source = 'fallback'
            fwd = np.mean([R[:, 2] for R in self.rotations], axis=0)
            fwd = np.array([fwd[0], fwd[1], 0.0])
            fwd /= np.linalg.norm(fwd)
            n = -fwd
            mid = c_mean + self.plane_distance * fwd
        if n @ (c_mean - mid) < 0:
            n = -n
        return n, float(n @ mid), source

    def dump_poses(self, K, n, d, grippers):
        '''Everything needed to reproduce the geometry offline.'''
        with open(self.sample_dir / 'poses.json', 'w') as f:
            json.dump({'world_frame': self.WORLD_FRAME, 'K_cropped': K.tolist(),
                       'D': self.D.tolist(), 'crop_top': self.CROP_TOP, 'crop_bottom': self.CROP_BOTTOM,
                       'rotations_cam_to_world': [np.asarray(R).tolist() for R in self.rotations],
                       'camera_centers': [np.asarray(c).tolist() for c in self.centers],
                       'plane_n': n.tolist(), 'plane_d': d, 'grippers': grippers}, f, indent=2)

    def make_panorama_plane(self):
        '''
        Builds the panorama with two geometric models and keeps the one whose images agree
        better in their overlaps:
          translation  - full TF poses (camera translation) + cloth plane from the grippers
          rotation_only - TF rotations only (as if the head rotated about the camera centre)
        Returns (panorama, refinement_applied, debug_dict).
        '''

        self.load_intrinsics()
        imgs = [self.preprocess(im) for im in self.images]
        masks = [self.valid_mask] * len(imgs)

        K = self.K.copy()
        K[1, 2] -= self.CROP_TOP          # principal point shifts after cropping

        Rs0, cs0 = list(self.rotations), [np.asarray(c) for c in self.centers]
        n0, d0, source = self.estimate_plane()
        shifts = [np.linalg.norm(a - b) for a in cs0 for b in cs0]
        debug = {'plane_source': source, 'plane_n': n0.tolist(), 'plane_d': d0,
                 'max_camera_shift_cm': float(100 * max(shifts)), 'candidates': {}}
        rospy.loginfo(f'cloth plane from {source}; camera moved up to {debug["max_camera_shift_cm"]:.1f} cm between shots')
        if self.save_debug:
            grip = {}
            for key, frame in (('left', self.left_gripper_frame), ('right', self.right_gripper_frame)):
                p = self.lookup(frame)
                grip[key] = None if p is None else p[1].tolist()
            self.dump_poses(K, n0, d0, grip)

        models = []
        if self.try_translation:
            models.append(('translation', cs0))
        if self.try_rotation_only or not models:
            models.append(('rotation_only', [np.mean(cs0, axis=0)] * len(cs0)))

        best = None
        for name, cs in models:
            Rs, n, d, applied, rinfo = list(Rs0), n0, d0, False, None
            if self.refine:
                try:
                    Rs2, n2, d2, rinfo = refine_poses(imgs, masks, Rs, cs, n, d, K,
                                                      max_corr_deg=self.max_corr_deg,
                                                      search_px=self.search_px, narrow_px=self.narrow_px,
                                                      log=lambda m, name=name: rospy.loginfo(f'[{name}] {m}'))
                    applied = bool(rinfo['applied'])
                    if applied:
                        Rs, n, d = Rs2, n2, d2
                except Exception as e:
                    rospy.logwarn(f'[{name}] pose refinement failed ({e}), using TF poses only')
            info = {}
            pano = stitch_on_plane(imgs, masks, Rs, cs, n, d, K, gain_comp=self.gain_comp, info=info)
            score = info['alignment_score']
            rospy.loginfo(f'[{name}] alignment score {score:.3f}, refinement {"applied" if applied else "NOT applied"}, '
                          f'size {pano.shape[1]}x{pano.shape[0]}')
            debug['candidates'][name] = {'alignment_score': score, 'refinement_applied': applied,
                                         'photometric_ab_bgr': info['photometric_ab_bgr'], 'refine': rinfo}
            if self.save_debug:
                cv2.imwrite(str(self.sample_dir / f'panorama_{self.sample_idx:04d}_{name}.png'), pano)
            if best is None or (np.nan_to_num(score, nan=-1.0) > np.nan_to_num(best[0], nan=-1.0)):
                best = (score, name, pano, applied)

        score, name, pano, applied = best
        debug['chosen_model'] = name
        rospy.loginfo(f'Chosen geometric model: {name} (score {score:.3f})')
        return pano, applied, debug

    def make_panorama_features(self):
        '''Feature-based panorama or None if it cannot be made completely and plausibly.'''
        h = self.images[0].shape[0]
        imgs = [im[self.CROP_TOP:h - self.CROP_BOTTOM] for im in self.images]
        return feature_stitch(imgs, log=rospy.loginfo)

    def make_panorama(self):

        pano_plane, applied, debug = None, False, {}
        if self.use_tf and all(R is not None for R in self.rotations):
            try:
                pano_plane, applied, debug = self.make_panorama_plane()
            except Exception as e:
                rospy.logwarn(f'Plane-based stitching failed ({e})')
        else:
            rospy.logwarn('Missing TF poses')

        if pano_plane is not None and (applied or not self.refine):
            method, pano = 'plane_refined' if applied else 'plane_tf_only', pano_plane
        else:
            # poses could not be verified by the images: see whether the feature stitcher can do better
            pano_feat = self.make_panorama_features()
            if pano_feat is not None:
                method, pano = 'features', pano_feat
            elif pano_plane is not None:
                rospy.logwarn('Using the TF-only plane panorama (not verified by image features)')
                method, pano = 'plane_tf_only', pano_plane
            else:
                raise RuntimeError('Panorama could not be created by any method')

        debug['method'] = method
        rospy.loginfo(f'Panorama method: {method}')
        if self.save_debug:
            with open(self.sample_dir / 'refinement.json', 'w') as f:
                json.dump(debug, f, indent=4)
        return pano

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
