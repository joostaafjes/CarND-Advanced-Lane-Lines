# Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from numpy.polynomial import Polynomial
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os
from itertools import product
import shutil
import datetime as dt
from PIL import Image

from matplotlib import pylab

from FindLines import FindLines
from SlidingWindowSearch import SlidingWindowSearch

# plt.interactive(False)

def undistort(img):
    # load pickle with mtx and dist
    dist_pickle = pickle.load(open("./camera_cal/wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst

# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(img, cv2.CV_64F, 1 if orient=='x' else 0, 1 if orient=='y' else 0, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img)  # Remove this line
    return sbinary

# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    sobelm = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * sobelm/ np.max(sobelm))
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sbinary


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx= np.absolute(sobelx)
    abs_sobely= np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    arctan = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    sbinary = np.zeros_like(arctan)
    sbinary[(arctan >= thresh[0]) & (arctan <= thresh[1])] = 1

    return sbinary

def hls_s(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]

    return s_channel

# Convert to HLS color space and separate the S channel
def hls_s_threshold(img, thresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 255

    return s_binary

def channel_threshold(img, thresh=(0,255)):
    # Threshold color channel
    s_binary = np.zeros_like(img)
    s_binary[(img >= thresh[0]) & (img <= thresh[1])] = 1

    return s_binary

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

start = dt.datetime.now()

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

image_files = [
    ['./straight_lines1', 'straight'],
    ['./straight_lines2', 'straight'],
    ['./test1', 'right'],
    ['./test2', 'left'],
    ['./test3', 'right'],
    ['./test4','right'],
    ['./test5', 'right'],
    ['./test6', 'right'],
]

images = []
findLines = FindLines(None)
# findLines = SlidingWindowSearch(None)

def process_image(input_img):
    # do all manipulation at undistorted images
    input_img = undistort(input_img)

    # HLS - S channel
    output_img_hls = cv2.cvtColor(input_img, cv2.COLOR_BGR2HLS)
    output_img_hls_2 = output_img_hls[:,:,2]
    output_img_hls_2_unwarped = unwarp_expand_top(output_img_hls_2)
    x_min_thresh = 20
    x_max_thresh = 255
    output_img_hls_2_unwarped_abs_sobel = abs_sobel_thresh(output_img_hls_2_unwarped, ksize, orient='x',thresh=(x_min_thresh, x_max_thresh))

    # LAB - B channel
    output_img_lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
    output_img_lab_2 = output_img_lab[:,:,2]
    output_img_lab_2_unwarped = unwarp_expand_top(output_img_lab_2)
    x_min_thresh = 20
    x_max_thresh = 255
    output_img_lab_2_unwarped_abs_sobel = abs_sobel_thresh(output_img_lab_2_unwarped, ksize, orient='x',thresh=(x_min_thresh, x_max_thresh))

    # RGB - R channel
    output_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    output_img_rgb_2 = output_img_rgb[:,:,0]
    output_img_rgb_2_unwarped = unwarp_expand_top(output_img_rgb_2)
    x_min_thresh = 20
    x_max_thresh = 255
    output_img_rgb_2_unwarped_abs_sobel = abs_sobel_thresh(output_img_rgb_2_unwarped, ksize, orient='x',thresh=(x_min_thresh, x_max_thresh))

    combined_or_binary = np.zeros_like(output_img_hls_2_unwarped_abs_sobel)
    combined_or_binary[(output_img_hls_2_unwarped_abs_sobel == 1) | (output_img_rgb_2_unwarped_abs_sobel == 1)] = 1

    findLines.warped = combined_or_binary
    result, msg = findLines.calculate()
    curve_left = 0
    curve_right = 0
    if True:
        if result:
            curve_left, curve_right = findLines.measure_curvation()
            msg += " - curves l : {} - r : {}".format(curve_left, curve_right)
        unwarped_lane = warp_shrink_top(findLines.draw_lines(unwarp_expand_top(input_img)))

        # use unwarped_lane as a mask to draw lane
        merged_image = input_img
        merged_image[unwarped_lane > 0] = [61, 148, 15]

        meters, txt = findLines.measure_center_difference()

        cv2.putText(merged_image,
                    'radius curvation l:{:4.0f} r:{:4.0f} avg:{:4.0f} m.-vehicle is {:.2f} m. {} of center'.format(curve_left, curve_right, (curve_right + curve_left) / 2, meters, txt),
                    (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        image_with_embeds = Image.fromarray(merged_image)

        if embed_intermediate_images:
        # HLS
            output_img_hls_2_unwarped_abs_sobel = output_img_hls_2_unwarped_abs_sobel * 255
            window_1 = Image.fromarray(output_img_hls_2_unwarped_abs_sobel)
            window_1.thumbnail((300, 300))
            image_with_embeds.paste(window_1, (10, 70))

            # RGB
            output_img_rgb_2_unwarped_abs_sobel = output_img_rgb_2_unwarped_abs_sobel * 255
            window_3 = Image.fromarray(output_img_rgb_2_unwarped_abs_sobel)
            window_3.thumbnail((300, 300))
            image_with_embeds.paste(window_3, (320, 70))

            combined_or_binary = combined_or_binary * 255
            window_3 = Image.fromarray(combined_or_binary)
            window_3.thumbnail((300, 300))
            image_with_embeds.paste(window_3, (630, 70))

            window_4 = Image.fromarray(findLines.out_img)
            window_4.thumbnail((300, 300))
            image_with_embeds.paste(window_4, (940, 70))

        # merged_image = input_img + unwarped_lane
        return np.array(image_with_embeds)

    return input_img


videos = [
    'project_video',
]

embed_intermediate_images = True

for input_file_name in videos:
    time_start = 0
    time_end = 50

    white_output = 'test_videos_output/' + input_file_name + "_%0.2f_%0.2f.mp4" % (time_start, time_end)
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("test_videos/" + input_file_name + ".mp4").subclip(time_start, time_end)

    clip1 = VideoFileClip("test_videos/" + input_file_name + ".mp4").subclip(time_start, time_end)
    white_output = 'test_videos_output/' + input_file_name + "_%0.2f_%0.2f.mp4" % (time_start, time_end)
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

