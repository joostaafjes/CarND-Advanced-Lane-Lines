import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os

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
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255

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
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255

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
    sbinary[(arctan >= thresh[0]) & (arctan <= thresh[1])] = 255

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
    s_binary[(img >= thresh[0]) & (img <= thresh[1])] = 255

    return s_binary

def unwarp_expand_top2(img):

    LEFT_BOTTOM_X = 230
    LEFT_BOTTOM_Y = 704
    RIGHT_BOTTOM_X = 1061
    RIGHT_BOTTOM_Y = 691
    LEFT_TOP_X = 568
    LEFT_TOP_Y = 469
    RIGHT_TOP_X = 748
    RIGHT_TOP_Y = 490
    src = np.float32([[LEFT_BOTTOM_X, LEFT_BOTTOM_Y], [LEFT_TOP_X, LEFT_TOP_Y],
                      [RIGHT_BOTTOM_X, RIGHT_BOTTOM_Y], [RIGHT_TOP_X, RIGHT_TOP_Y]])
    dst = np.float32([[LEFT_BOTTOM_X, LEFT_BOTTOM_Y], [LEFT_BOTTOM_X, LEFT_TOP_Y],
                      [RIGHT_BOTTOM_X, RIGHT_BOTTOM_Y], [RIGHT_BOTTOM_X, RIGHT_TOP_Y]])
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size)

    return warped


def unwarp_expand_top(img):

    DST_MARGIN_X = 100

    TOP_Y = 450
    LEFT_TOP_X = 596
    RIGHT_TOP_X = 684

    BOTTOM_Y = 675
    LEFT_BOTTOM_X = 270
    RIGHT_BOTTOM_X = 1038

    src = np.float32([[LEFT_BOTTOM_X, BOTTOM_Y], [LEFT_TOP_X, TOP_Y],
                      [RIGHT_BOTTOM_X, BOTTOM_Y], [RIGHT_TOP_X, TOP_Y]])
    dst = np.float32([[LEFT_BOTTOM_X + DST_MARGIN_X, 720], [LEFT_BOTTOM_X + DST_MARGIN_X, 0],
                      [RIGHT_BOTTOM_X - DST_MARGIN_X, 720], [RIGHT_BOTTOM_X - DST_MARGIN_X, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size)

    return warped

def unwarp_shrink_bottom(img):

    LEFT_BOTTOM_X = 230
    LEFT_BOTTOM_Y = 704
    RIGHT_BOTTOM_X = 1061
    RIGHT_BOTTOM_Y = 691
    LEFT_TOP_X = 568
    LEFT_TOP_Y = 469
    RIGHT_TOP_X = 748
    RIGHT_TOP_Y = 490
    src = np.float32([[LEFT_BOTTOM_X, LEFT_BOTTOM_Y], [LEFT_TOP_X, LEFT_TOP_Y],
                      [RIGHT_BOTTOM_X, RIGHT_BOTTOM_Y], [RIGHT_TOP_X, RIGHT_TOP_Y]])
    dst = np.float32([[LEFT_TOP_X, LEFT_BOTTOM_Y], [LEFT_TOP_X, LEFT_TOP_Y],
                      [RIGHT_TOP_X, RIGHT_BOTTOM_Y], [RIGHT_TOP_X, RIGHT_TOP_Y]])
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size)

    return warped

def unwarp_shrink_bottom(img):
    IMAGE_H = 223
    IMAGE_W = 1280

    src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src)  # Inverse transformation

    img = img[450:(450 + IMAGE_H), 0:IMAGE_W]  # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))  # Image warping

    return warped_img

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


# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
# gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
# grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
# mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
# dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))


# combined = np.zeros_like(dir_binary)
# combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

images = [
    ['./straight_lines1', 'straight'],
    ['./straight_lines2', 'straight'],
    ['./test1', 'right'],
    ['./test2', 'left'],
    ['./test3', 'right'],
    ['./test4','right'],
    ['./test5', 'right'],
    ['./test6', 'right'],
]

findLines = FindLines(None)
# findLines = SlidingWindowSearch(None)
for image, expected_corner in images:

    input_img_bgr = cv2.imread('./test_images/' + image + '.jpg')
    if input_img_bgr is None:
        print('Error reading file ' + image)
        exit(1)
    else:
        print('Start processing image ' + image + '...')


# COLOR_BGR2GRAY = 6
# COLOR_BGR2HLS = 52
# COLOR_BGR2HSV = 40
# COLOR_BGR2LAB = 44

    # for min_thresh in range(0, 200, 20):
    #     for max_thresh in range(min_thresh + 10, 255, 20):
    #         for ksize in range(3, 15, 2):
    # for colorspace in [cv2.COLOR_BGR2GRAY, cv2.COLOR_BGR2HLS, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2LAB]:
    for colorspace in [cv2.COLOR_BGR2HLS]:
        print('start colorspace {}'.format(colorspace))
        # for color_channel in [0, 1, 2]:
        for color_channel in [2]:
            print('start channel {}'.format(color_channel))
            # for method in ['normal', 'sobel_x', 'sobel_y', 'sobel_mag', 'sobel_dir']:
            for method in ['sobel_dir', ]:
                print('start method {}'.format(method))
                if method == 'sobel_dir':
                    # min_thresh_range = np.arange(1.0, np.pi / 2, 0.1)
                    # max_thresh_range = np.arange(1.4, np.pi / 2, 0.1)
                    min_thresh_range = np.arange(0, np.pi / 2, 0.1)
                    max_thresh_range = np.arange(0, np.pi / 2, 0.1)
                else:
                    min_thresh_range = range(90, 100, 20)
                    max_thresh_range = range(255, 260, 20)
                    # min_thresh_range = range(5, 260, 50)
                    # max_thresh_range = range(5, 260, 50)
                if method == 'normal':
                    ksize_range = range(13, 15, 2)
                else:
                    ksize_range = range(3, 5, 2)
                dir_name = './test_images/out/' + str(colorspace) + '/' + str(color_channel) + '/' + method + '/'
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                for min_thresh in min_thresh_range:
                    for max_thresh in max_thresh_range:
                        if min_thresh > max_thresh:
                            continue
                        for ksize in ksize_range:
                            file_name = '{}_{:3.1f}_{:3.1f}_{}'.format(image, min_thresh, max_thresh, ksize)

                            # cv2.imwrite(dir_name + file_name  + '_a.jpg', unwarp_expand_top(input_img_bgr))

                            input_img = cv2.cvtColor(input_img_bgr, colorspace)
                            output_img = undistort(input_img)
                            # cv2.imwrite(dir_name + file_name  + '_0.jpg', output_img)

                            if colorspace != cv2.COLOR_BGR2GRAY:
                                output_img= output_img[:,:,color_channel]

                            # test which is best
                            # cv2.imwrite(dir_name + file_name  + '_1a.jpg', channel_threshold(unwarp_expand_top(output_img), thresh=(min_thresh, max_thresh)))
                            # cv2.imwrite(dir_name + file_name  + '_1b.jpg', unwarp_expand_top(channel_threshold(output_img, thresh=(min_thresh, max_thresh))))

                            output_img = unwarp_expand_top(output_img)
                            # cv2.imwrite(dir_name + file_name  + '_2.jpg', output_img)

                            if method == 'normal':
                                output_img = channel_threshold(output_img, thresh=(min_thresh, max_thresh))
                            elif method == 'sobel_x':
                                output_img = abs_sobel_thresh(output_img, ksize, orient='x',thresh=(min_thresh, max_thresh))
                            elif method == 'sobel_y':
                                output_img = abs_sobel_thresh(output_img, ksize, orient='y',thresh=(min_thresh, max_thresh))
                            elif method == 'sobel_mag':
                                output_img = mag_thresh(output_img, ksize, thresh=(min_thresh, max_thresh))
                            elif method == 'sobel_dir':
                                output_img = dir_threshold(output_img, ksize, thresh=(min_thresh, max_thresh))
                            cv2.imwrite(dir_name + file_name  + '_3.jpg', output_img)

                            output_img = region_of_interest(output_img)
                            # cv2.imwrite(dir_name + file_name  + '_2.jpg', output_img)

                            # output_img = hls_s(output_img)
                            # output_img_binary = mag_thresh(output_img, sobel_kernel=ksize, thresh=(min_thresh, max_thresh))
                            # output_img = np.zeros_like(output_img_binary)
                            # output_img[(output_img_binary == 1)] = 255
                            # cv2.imwrite('./test_images/out/' + image + '_' + str(min_thresh) + '_' + str(max_thresh) + '_' + str(ksize) + '_2.jpg', output_img)
                            # output_img = hls_s_threshold(output_img, (60, 230))

                            # cv2.imwrite(dir_name + file_name  + '_3a.jpg', unwarp_expand_top(output_img))
                            # cv2.imwrite(dir_name + file_name  + '_3b.jpg', unwarp_shrink_bottom(output_img))
                            # cv2.imwrite(dir_name + file_name  + '_3c.jpg', unwarp_shrink_bottom2(output_img))

                            # print("thresh {}-{}".format(min_thresh, max_thresh))
                            findLines.warped = output_img
                            result, msg = findLines.calculate(expected_corner)
                            if result:
                                result, curve_left, curve_right = findLines.measure_curvation()
                                if result:
                                    msg += " - curves l : {} - r : {}".format(curve_left, curve_right)
                                # Plotting thresholded images
                                # ax1.set_title('Stacked thresholds')
                                # ax1.imshow(cv2.cvtColor(input_img_bgr, cv2.COLOR_BGR2RGB))
                                    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

                                    histogram = np.sum(output_img[output_img.shape[0] // 2:, :], axis=0)
                                    ax1.imshow(output_img, cmap='gray')
                                    ax2.plot(histogram)
                                    ax2.set_title('Histogram')


                                # print(output_img.shape[0])
                                    findLines.plot(ax1, msg)

                                # print("{} - curve l {:6.0f} - curve r {:6.0f} for thresh {}-{}, kernel {}".format(image, findLines.left_curverad, findLines.right_curverad, min_thresh, max_thresh, ksize))
                                # ax2.set_title("{} - curve l {:6.0f} - curve r {:6.0f} for thresh {}-{}, kernel {}".format(image, findLines.left_curverad, findLines.right_curverad, min_thresh, max_thresh, ksize))
                                # ax2.imshow(output_img)

                                # plt.show(block=False)
                                    figure.savefig(dir_name + file_name + '_4.png')
                                    plt.close(figure)

                findLines.reset_curv_diff()



