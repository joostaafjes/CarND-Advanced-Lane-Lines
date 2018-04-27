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
    LEFT_TOP_X = 593
    RIGHT_TOP_X = 690

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

def region_of_interest(img):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    top_margin = 280
    lower_margin = 50
    img_shape = output_img.shape
    min_y = img_shape[0] - top_margin
    max_y = img_shape[0] - lower_margin
    max_x = img_shape[1]
    vertices = np.array([[(0, max_y), (0, min_y), (max_x, min_y), (max_x, max_y)]], dtype=np.int32)

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def bin_to_img(img_binary):
    output_img = np.zeros_like(img_binary)
    output_img[(img_binary == 1)] = 255
    return output_img

def combine(*s_binary):
    # Combine the multiple binary thresholds
    combined_binary = np.zeros_like(s_binary[0])
    combined_binary[(s_binary == 1)] = 1
    for index in range(len(s_binary) - 1):
        combined_binary[(combined_binary == 1) | (s_binary[index + 1] == 1)] = 1

    return combined_binary

def combine_methods_and_write(*methods):
    filename = dir_name + file_name + '_'
    combined_binary = np.array([])
    for method in methods:
        filename += method
        if method == 'n':
            s_binary = normal_bin_output_img
        elif method == 'x':
            s_binary = sobelx_bin_output_img
        elif method == 'y':
            s_binary = sobely_bin_output_img
        elif method == 'm':
            s_binary = sobelm_bin_output_img
        else:
            s_binary = sobeld_bin_output_img

        # Combine the multiple binary thresholds
        if combined_binary.size == 0:
            combined_binary = np.zeros_like(s_binary)
            combined_binary[(s_binary == 1)] = 1
        else:
            combined_binary[(combined_binary == 1) & (s_binary == 1)] = 1

    filename += ".jpg"
    cv2.imwrite(filename, bin_to_img(combined_binary))

def combine_methods(combinations, write=False):
    filename = dir_name + file_name + '_' + combinations
    combined_binary = np.array([])

    if combinations[0] == 'n':
        s_binary = normal_bin_output_img
    elif combinations[0] == 'x':
        s_binary = sobelx_bin_output_img
    elif combinations[0] == 'y':
        s_binary = sobely_bin_output_img
    elif combinations[0] == 'm':
        s_binary = sobelm_bin_output_img
    else:
        s_binary = sobeld_bin_output_img
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_binary == 1)] = 1

    if len(combinations) > 1:
        for index in range((len(combinations) - 1) // 2):
            method = combinations[(index + 1) * 2]
            if method == 'n':
                s_binary = normal_bin_output_img
            elif method == 'x':
                s_binary = sobelx_bin_output_img
            elif method == 'y':
                s_binary = sobely_bin_output_img
            elif method == 'm':
                s_binary = sobelm_bin_output_img
            else:
                s_binary = sobeld_bin_output_img
            if combinations[index * 2 + 1] == '&':
                combined_binary[(combined_binary == 1) & (s_binary == 1)] = 1
            else:
                combined_binary[(combined_binary == 1) | (s_binary == 1)] = 1

    filename += ".jpg"
    img = bin_to_img(combined_binary)
    if write:
        cv2.imwrite(filename, img)

    return img

def get_all_combinations_old():
    n = 9
    count = 0
    mylist = []
    for combination in range(1, 2 ** n):
        comb = ''
        if combination & (2 ** (n - 1)) > 0:
            comb += 'n'
        if combination & (2 ** (n - 3)) > 0:
            if comb != '':
                if combination & (2 ** (n - 2)) > 0:
                    comb += '&'
                else:
                    comb += '|'
            comb += 'x'
        if combination & (2 ** (n - 5)) > 0:
            if comb != '':
                if combination & (2 ** (n - 4)) > 0:
                    comb += '&'
                else:
                    comb += '|'
            comb += 'y'
        if combination & (2 ** (n - 7)) > 0:
            if comb != '':
                if combination & (2 ** (n - 6)) > 0:
                    comb += '&'
                else:
                    comb += '|'
            comb += 'm'
        if combination & (2 ** (n - 9)) > 0:
            if comb != '':
                if combination & (2 ** (n - 8)) > 0:
                    comb += '&'
                else:
                    comb += '|'
            comb += 'd'
        if comb != '':
            mylist.append(comb)
            count += 1
    myset = set(mylist)
    mylist2 = list(myset)

    return mylist2

def get_all_combinations(length=1):

    methods = ['n', 'x', 'y', 'm', 'd', '']
    oper = ['|', '&', '']

    a = product(methods, oper, methods, oper, methods, oper, methods, oper, methods)

    def f(x):
        z = []
        for y in filter(str.isalpha, x):
            z.append(y)
        return len(z) == len(set(z))

    # remove double letters
    it = filter(f, a)

    mylist = []
    for i in it:
        txt = i[0] + i[1] + i[2] + i[3] + i[4] + i[5] + i[6] + i[7] + i[8]
        if len(txt) == 0:
            continue
        if len(txt) != length:
            continue
        if not str.isalpha(txt[0]) or not str.isalpha(txt[-1]):
            continue
        if len(txt) > 1 and str.isalpha(txt):
            continue
        if len(txt) % 2 == 0:
            continue
        skip = False
        if len(txt) > 1:
            for index in range(len(txt) - 1):
                if str.isalpha(txt[index]) and str.isalpha(txt[index + 1]):
                    skip = True
                    break
                if not str.isalpha(txt[index]) and not str.isalpha(txt[index + 1]):
                    skip = True
                    break
        if skip:
            continue
        if txt[0] > txt[-1]:
            txt = txt[::-1]
        mylist.append(txt)

    mylist = sorted(list(set(mylist)))
    print('# of combinations:', len(mylist))

    return mylist

def get_string_for_combination(combination):
    result = ''
    if 'n' in combination:
        result += 'n{:3.1f}_{:3.1f}'.format(n_min_thresh, n_max_thresh)
    if 'x' in combination:
        result += 'x{:3.1f}_{:3.1f}'.format(x_min_thresh, x_max_thresh)
    if 'y' in combination:
        result += 'y{:3.1f}_{:3.1f}'.format(y_min_thresh, y_max_thresh)
    if 'm' in combination:
        result += 'm{:3.1f}_{:3.1f}'.format(m_min_thresh, m_max_thresh)
    if 'd' in combination:
        result += 'd{:3.1f}_{:3.1f}'.format(sobel_dir_min_thresh, sobel_dir_max_thresh)
    result += '_k{:d}'.format(ksize)
    return result

start = dt.datetime.now()

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# combined = np.zeros_like(dir_binary)
# combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

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

colorspace = cv2.COLOR_BGR2HLS
color_channel = 2

def process_image(input_img):
    output_img = cv2.cvtColor(input_img, colorspace)
    # cv2.imwrite(dir_name + file_name  + '_0.jpg', output_img)

    if colorspace != cv2.COLOR_BGR2GRAY:
        output_img= output_img[:,:,color_channel]

    output_img = unwarp_expand_top(output_img)
    # cv2.imwrite(dir_name + file_name  + '_2.jpg', output_img)

    x_min_thresh = 20
    x_max_thresh = 255
    output_img = abs_sobel_thresh(output_img, ksize, orient='x',thresh=(x_min_thresh, x_max_thresh))

    if not output_img.any():
        print('skip all 0')

    findLines.warped = output_img
    result, msg = findLines.calculate()
    if True:
        if result:
            result, curve_left, curve_right, curve_div = findLines.measure_curvation()
            msg += " - curves l : {} - r : {}".format(curve_left, curve_right)
        # figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # histogram = np.sum(output_img[output_img.shape[0] // 2:, :], axis=0)
        # ax1.imshow(output_img, cmap='gray')
        # ax1.set_title(msg)
        # ax2.plot(histogram)
        # ax2.set_title('Histogram')

        # output_img = findLines.plot(ax1, title=msg, input_image=input_img)

        # return warp_shrink_top(findLines.draw_lines(unwarp_expand_top(input_img)))
        unwarped_lane = warp_shrink_top(findLines.draw_lines(unwarp_expand_top(input_img)))

        merged_image = input_img
        merged_image[unwarped_lane > 0] = 255

        # merged_image = input_img + unwarped_lane
        return merged_image

        # plt.close(figure)

    return input_img


videos = [
    # 'challenge',
    'harder_challenge_video',
    # 'solidWhiteRight',
    'challenge_video',
    'project_video',
    # 'solidYellowLeft',
]

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

