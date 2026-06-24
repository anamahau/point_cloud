#!/usr/bin/env python

import cv2
import json
import rospy
import shutil
import imutils
import argparse
import numpy as np
from imutils import paths


def find_matches_old(img1, img2):
    # orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    knn_matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []

    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)

    return kp1, kp2, good_matches


def find_matches(img1, img2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return kp1, kp2, []

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50)
    )

    knn = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in knn:
        if m.distance < 0.65 * n.distance:
            good.append(m)

    good = sorted(good, key=lambda x: x.distance)

    return kp1, kp2, good


def estimate_homography(kp1, kp2, matches):

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)

    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    return H, mask


def estimate_transform(kp1, kp2, matches, use_affine=True):

    if (len(matches) < 10):
        return None, None
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    if use_affine:
        M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if M is None:
            return None, None
        H = np.vstack([M, [0, 0, 1]])
        return H, mask
    
    else:
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        return H, mask


def compute_H(img_src, img_ref):
    kp1, kp2, matches = find_matches(img_src, img_ref)
    H, mask = estimate_transform(kp1, kp2, matches)

    if H is None:
        return RuntimeError('Could not compute transform')
    
    return H


def create_canvas(img2, scale=3):
    h2, w2 = img2.shape[:2]

    canvas_h = h2 * scale
    canvas_w = w2 * scale

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    offset_x = w2
    offset_y = h2

    canvas[offset_y:offset_y+h2, offset_x:offset_x+w2] = img2

    T = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float32)

    return canvas, T


def warp(img, H, shape):
    return cv2.warpPerspective(
        img,
        H,
        (shape[1], shape[0]),
        flags=cv2.INTER_NEAREST
    )


def draw_matches(img1, img2):
    kp1, kp2, matches = find_matches(img1, img2)
    H, mask = estimate_homography(
        kp1,
        kp2,
        matches
    )
    inlier_matches = [
        matches[i]
        for i in range(len(matches))
        if mask[i]
    ]
    vis = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        inlier_matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return inlier_matches, vis


def stitch_three_new(img1, img2, img3, depth1=None, depth2=None, depth3=None):

    canvas, T = create_canvas(img2)

    H12 = compute_H(img1, img2)
    H32 = compute_H(img3, img2)

    H12c = T @ H12
    H32c = T @ H32

    h, w = canvas.shape[:2]

    warp1 = warp(img1, H12c, (h, w))
    warp3 = warp(img3, H32c, (h, w))

    canvas_rgb = canvas.copy()

    mask_canvas = np.any(canvas_rgb > 0, axis=2)

    m1 = np.any(warp1 > 0, axis=2)
    canvas_rgb[m1 & (~mask_canvas)] = warp1[m1 & (~mask_canvas)]

    mask_canvas = np.any(canvas_rgb > 0, axis=2)

    m3 = np.any(warp3 > 0, axis=2)
    canvas_rgb[m3 & (~mask_canvas)] = warp3[m3 & (~mask_canvas)]

    canvas_depth = None

    if depth1 is not None:
        canvas_depth = np.zeros((h, w), dtype=depth1.dtype)

        d2 = depth2 if depth2 is not None else np.zeros_like(depth1)

        canvas_depth[
            canvas.shape[0]//2 : canvas.shape[0]//2 + d2.shape[0],
            canvas.shape[1]//2 : canvas.shape[1]//2 + d2.shape[1]
        ] = d2

        d1 = warp(depth1, H12c, (h, w))
        d3 = warp(depth3, H32c, (h, w))

        m1 = d1 > 0
        m3 = d3 > 0

        canvas_depth[m1] = d1[m1]
        canvas_depth[m3] = d3[m3]

    return canvas_rgb, canvas_depth, (H12, H32)


def stitch_three(img1, img2, img3, depth1=None, depth2=None, depth3=None):

    h2, w2 = img2.shape[:2]

    # canvas_h = h2 * 3
    # canvas_w = w2 * 2
    canvas_h = 1024
    canvas_w = 1024

    canvas_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # offset_x = w2 // 2
    # offset_y = h2
    offset_x = 192
    offset_y = 272

    blurred1 = maskFromDepth(depth1, img1)
    blurred2 = maskFromDepth(depth2, img2)
    blurred3 = maskFromDepth(depth3, img3)

    canvas_rgb[offset_y:offset_y+h2, offset_x:offset_x+w2] = blurred2

    T = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ])

    # img1 -> img2
    # kp1, kp2, matches12 = find_matches(img1, img2)
    kp1, kp2, matches12 = find_matches(blurred1, blurred2)
    H12, _ = estimate_homography(kp1, kp2, matches12)
    # print(f'H12 = \n{H12}')

    # img3 -> img2
    # kp3, kp2b, matches32 = find_matches(img3, img2)
    kp3, kp2b, matches32 = find_matches(blurred3, blurred2)
    H32, _ = estimate_homography(kp3, kp2b, matches32)
    # print(f'H23 = \n{H32}')

    if H32 is None:
        H32 = np.eye(3, dtype=np.float32)
    if H12 is None:
        H12 = np.eye(3, dtype=np.float32)

    warp1 = cv2.warpPerspective(blurred1, T @ H12, (canvas_w, canvas_h))
    warp3 = cv2.warpPerspective(blurred3, T @ H32, (canvas_w, canvas_h))

    mask_canvas = np.any(canvas_rgb > 0, axis=2)
    mask1 = np.any(warp1 > 0, axis=2)
    fill1 = mask1 & (~mask_canvas)
    canvas_rgb[fill1] = warp1[fill1]

    mask_canvas = np.any(canvas_rgb > 0, axis=2)
    mask3 = np.any(warp3 > 0, axis=2)
    fill3 = mask3 & (~mask_canvas)
    canvas_rgb[fill3] = warp3[fill3]

    if depth1 is None:
        return canvas_rgb, None
    
    else:
        canvas_depth = np.zeros((canvas_h, canvas_w), dtype=depth1.dtype)
        canvas_depth[offset_y:offset_y+h2, offset_x:offset_x+w2] = depth2
        warp1_d = cv2.warpPerspective(depth1, T @ H12, (canvas_w, canvas_h))
        warp3_d = cv2.warpPerspective(depth3, T @ H32, (canvas_w, canvas_h))
        mask1_d = warp1_d > 0
        mask3_d = warp3_d > 0
        fill1_d = mask1_d & (canvas_depth == 0)
        fill3_d = mask3_d & (canvas_depth == 0)
        canvas_depth[fill1_d] = warp1_d[fill1_d]
        canvas_depth[fill3_d] = warp3_d[fill3_d]
        return canvas_rgb, canvas_depth


def copySample(number, rgb, depth):
    shutil.copytree(f'/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference/sample_{(number+1):06d}', f'/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference/sample_{(number):06d}_{(number+2):06d}')

    cameraJson = f'/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference/sample_{(number):06d}_{(number+2):06d}/observation_start/camera_intrinsics.json'
    with open(cameraJson, 'r') as f:
        data = json.load(f)
    data["image_resolution"]["width"] = 1280
    data["image_resolution"]["height"] = 1440
    with open(cameraJson, 'w') as f:
        json.dump(data, f, indent=4)

    modelJson = f'/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference/sample_{(number):06d}_{(number+2):06d}/observation_start/requested_model.json'
    with open(modelJson, 'r') as f:
        data = json.load(f)
    data["requested_model"] = "v2+seg+rand_bg+cropping_talos"
    with open(modelJson, 'w') as f:
        json.dump(data, f, indent=4)
    
    path = f'/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference/sample_{(number):06d}_{(number+2):06d}/observation_start/'

    rgb_small = cv2.resize(rgb, (768, 768))
    rgb_padded = cv2.copyMakeBorder(
        rgb_small,
        top=0,
        bottom=256,
        left=128,
        right=128,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    depth_small = cv2.resize(depth, (768, 768))
    depth_padded = cv2.copyMakeBorder(
        depth_small,
        top=0,
        bottom=256,
        left=128,
        right=128,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    
    cv2.imwrite(
        path + 'image_left.png',
        rgb_padded
    )
    cv2.imwrite(
        path + 'depth_map.tiff',
        depth_padded
    )

    shutil.rmtree(f'/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference/sample_{(number):06d}_{(number+2):06d}/grasp_predicted/')


def copySample_1img(number):
    shutil.copytree(f'/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference/sample_{(number):06d}', f'/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference/sample_{(number):06d}_01')

    cameraJson = f'/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference/sample_{(number):06d}_01/observation_start/camera_intrinsics.json'
    with open(cameraJson, 'r') as f:
        data = json.load(f)
    data["image_resolution"]["width"] = 1024
    data["image_resolution"]["height"] = 1024
    with open(cameraJson, 'w') as f:
        json.dump(data, f, indent=4)

    rgb = cv2.imread(f'/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference/sample_{(number):06d}_01/observation_start/image_left.png')
    depth = cv2.imread(f'/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference/sample_{(number):06d}_01/observation_start/depth_map.tiff', cv2.IMREAD_UNCHANGED)

    modelJson = f'/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference/sample_{(number):06d}_01/observation_start/requested_model.json'
    with open(modelJson, 'r') as f:
        data = json.load(f)
    data["requested_model"] = "v2+seg+rand_bg+cropping_talos"
    with open(modelJson, 'w') as f:
        json.dump(data, f, indent=4)
    
    path = f'/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference/sample_{(number):06d}_01/observation_start/'

    rgb_small = cv2.resize(rgb, (480, 360))
    rgb_padded = cv2.copyMakeBorder(
        rgb_small,
        top=100,
        bottom=564,
        left=272,
        right=272,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    depth_small = cv2.resize(depth, (480, 360))
    depth_padded = cv2.copyMakeBorder(
        depth_small,
        top=100,
        bottom=564,
        left=272,
        right=272,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    
    cv2.imwrite(
        path + 'image_left.png',
        rgb_padded
    )
    cv2.imwrite(
        path + 'depth_map.tiff',
        depth_padded
    )

    shutil.rmtree(f'/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference/sample_{(number):06d}_01/grasp_predicted/')


def fillHoles(mask):
    h, w = mask.shape
    inv = cv2.bitwise_not(mask)
    flood = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = mask | flood_inv
    return filled


def maskFromDepth(depthImg, rgbImg, expand_px=30, blur=51):
    mask = ((depthImg > 0.0) & (depthImg < 0.8)).astype(np.uint8) * 255
    kernel1 = np.ones((expand_px, expand_px), np.uint8)
    mask = cv2.dilate(mask, kernel1, iterations=1)
    mask_bin = mask > 0
    rgb_masked = rgbImg.copy()
    rgb_masked[~mask_bin] = 0
    blurred = cv2.GaussianBlur(rgbImg, (blur, blur), 0)
    mask_3c = mask_bin[..., None]
    rgb_result = np.where(mask_3c, rgbImg, blurred)
    return rgbImg


if __name__ == '__main__':

    rospy.init_node('panorama_image_node', anonymous=True)

    i = 151

    images = []
    depths = []
    for j in range(3):
        image = cv2.imread(f'/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference/sample_{(i+j):06d}/observation_start/image_left.png')
        images.append(image)
        depth = cv2.imread(f'/home/anamarija/cedirnet-dev/tools/unfolding_evaluation_ral2025/grasp_inference/sample_{(i+j):06d}/observation_start/depth_map.tiff', cv2.IMREAD_UNCHANGED)
        depths.append(depth)

    img1 = images[0]
    img2 = images[1]
    img3 = images[2]
    depth1 = depths[0]
    depth2 = depths[1]
    depth3 = depths[2]

    # rgb_panorama, depth_panorama, Hs = stitch_three_new(img1, img2, img3)
    rgb_panorama, depth_panorama = stitch_three(img1, img2, img3, depth1, depth2, depth3)

    copySample(i, rgb_panorama, depth_panorama)
    # copySample_1img(140)

    # maskFromDepth(depth2, img2)