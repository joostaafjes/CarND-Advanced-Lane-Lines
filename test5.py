import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle

img = cv2.imread('./test_images/straight_lines1.jpg')

def undistort(img):
    # load pickle with mtx and dist
    dist_pickle = pickle.load(open("./camera_cal/wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst

def get_perspective_parameters():
    DST_MARGIN_X = 100

    TOP_Y = 450
    LEFT_TOP_X = 596
    RIGHT_TOP_X = 685

    BOTTOM_Y = 675
    LEFT_BOTTOM_X = 270
    RIGHT_BOTTOM_X = 1038

    src = np.float32([[LEFT_BOTTOM_X, BOTTOM_Y], [LEFT_TOP_X, TOP_Y],
                      [RIGHT_BOTTOM_X, BOTTOM_Y], [RIGHT_TOP_X, TOP_Y]])
    dst = np.float32([[LEFT_BOTTOM_X + DST_MARGIN_X, 720], [LEFT_BOTTOM_X + DST_MARGIN_X, 0],
                      [RIGHT_BOTTOM_X - DST_MARGIN_X, 720], [RIGHT_BOTTOM_X - DST_MARGIN_X, 0]])

    return src, dst

def unwarp_expand_top(img):
    src, dst = get_perspective_parameters()
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    unwarped = cv2.warpPerspective(img, M, img_size)

    return unwarped

def warp_shrink_top(img):
    src, dst = get_perspective_parameters()
    M = cv2.getPerspectiveTransform(dst, src)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size)

    return warped

out_img = undistort(img)

out_img = unwarp_expand_top(out_img)

cv2.line(out_img, (372, 0), (372, 720), (0, 0, 255), thickness=5)
cv2.line(out_img, (936, 0), (936, 720), (0, 0, 255), thickness=5)

cv2.imwrite('./test_images/straight_lines1_unwarp.jpg', out_img)
