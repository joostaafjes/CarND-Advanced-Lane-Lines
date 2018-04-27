import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure



class FindLines:

    def __init__(self, warped):
        self.warped = warped
        self.first = True
        self.reset_curv_diff()

    def calculate(self, expected_corner=None):
        if self.first:
            return self.__calculate(expected_corner)
        else:
            return self.__calculate_known(expected_corner)

    def plot(self, ax, title='', input_image=None):
        if self.first:
            self.__plot(ax, title, input_image)
        else:
            self.__plot_known(ax, title)
        # self.first = False

    def __calculate(self, expected_corner):
        # Assuming you have created a warped binary image called "warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(self.warped[self.warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result (convert to RGB with dstack)
        self.out_img = np.dstack((self.warped, self.warped, self.warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # check if valid
        center_left = 363
        margin_left = 100
        if leftx_base < (center_left - margin_left) or leftx_base > (center_left+ margin_left):
            return False, 'Left lane start outside expected centre {}'.format(leftx_base)
        center_right = 937
        margin_right = 100
        if rightx_base < (center_right - margin_right) or rightx_base > (center_right+ margin_right):
            return False, 'Right lane start outside expected centre {}'.format(rightx_base)

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(self.warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.warped.shape[0] - (window+1)*window_height
            win_y_high = self.warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(self.out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2)
            cv2.rectangle(self.out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
            (self.nonzerox >= win_xleft_low) &  (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
            (self.nonzerox >= win_xright_low) &  (self.nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            self.left_lane_inds.append(good_left_inds)
            self.right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        self.left_lane_inds = np.concatenate(self.left_lane_inds)
        self.right_lane_inds = np.concatenate(self.right_lane_inds)

        # Extract left and right line pixel positions
        self.leftx = self.nonzerox[self.left_lane_inds]
        self.lefty = self.nonzeroy[self.left_lane_inds]
        self.rightx = self.nonzerox[self.right_lane_inds]
        self.righty = self.nonzeroy[self.right_lane_inds]

        # Fit a second order polynomial to each
        if len(self.leftx) == 0 or len(self.lefty) == 0 or len(self.rightx) == 0 or len(self.righty) == 0:
            return False
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)

        # Generate x and y values for plotting
        self.ploty = np.linspace(0, self.warped.shape[0] - 1, self.warped.shape[0])
        self.left_fitx = self.left_fit[0] * self.ploty ** 2 + self.left_fit[1] * self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0] * self.ploty ** 2 + self.right_fit[1] * self.ploty + self.right_fit[2]

        if expected_corner:
            if (self.left_fitx.max() - self.left_fitx.min()) > 25 and expected_corner == 'straight':
                return False,  'left lane not straight'

            if (self.right_fitx.max() - self.right_fitx.min()) > 25 and expected_corner == 'straight':
                return False, 'right lane not straight'

            d1 = self.left_fitx[1:] - self.left_fitx[:-1]
            a1 = np.average(self.left_fitx[1:] - self.left_fitx[:-1])
            d2 = self.right_fitx[1:] - self.right_fitx[:-1]
            a2 = np.average(self.right_fitx[1:] - self.right_fitx[:-1])
            if expected_corner == 'right' and np.average(self.left_fitx[1:] - self.left_fitx[:-1]) > -0.1:
                return False, 'left lane not right curve '

            if expected_corner == 'right' and np.average(self.right_fitx[1:] - self.right_fitx[:-1]) > -0.1:
                return False, 'right lane not right curve '

            if expected_corner == 'left' and np.average(self.left_fitx[1:] - self.left_fitx[:-1]) < 0.1:
                return False, 'left lane not left curve '

            if expected_corner == 'left' and np.average(self.right_fitx[1:] - self.right_fitx[:-1]) < 0.1:
                return False, 'right lane not left curve '

        return True, 'curves as expected'

    def __plot(self, ax, title, input):
        if input.any() == None:
            self.out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
            self.out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]
            ax.imshow(self.out_img)
        else:
            ax.imshow(input)
        ax.plot(self.left_fitx, self.ploty, color='red')
        ax.plot(self.right_fitx, self.ploty, color='red')
        ax.set_xlim(0, 1280)
        ax.set_ylim(720, 0)
        ax.set_title(title)

            # fig = Figure()
            # canvas = FigureCanvas(fig)
            # ax = fig.gca()
            #
            # ax.imshow(input)
            # ax.plot(self.left_fitx, self.ploty, color='yellow')
            # ax.plot(self.right_fitx, self.ploty, color='yellow')
            #
            # canvas.draw()  # draw the canvas, cache the renderer
            # width, height = fig.get_size_inches() * fig.get_dpi()
            # width = 1280
            # height = 720
            # output_image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            #
            # return output_image
        if input.any() == None:
            return input
        else:
            return self.out_img

    def draw_lines(self, input_image):
        """
        draw marked lane at black background with same size as input_image
        :param input_image:
        :return black image with lane marked:
        """
        background_image = np.zeros_like(input_image)
        left_line_poly = np.array((self.left_fitx, self.ploty)).T.astype(np.int32)
        left_line_poly = left_line_poly.reshape((-1,1,2))
        # cv2.polylines(input_image, [left_line_poly], False, (0,255,255), thickness=10)

        right_line_poly = np.array((self.right_fitx, self.ploty)).T.astype(np.int32)
        right_line_poly = right_line_poly.reshape((-1,1,2))

        combined_line = np.append(left_line_poly, right_line_poly[::-1])
        combined_line = combined_line.reshape((-1, 1, 2))

        cv2.fillPoly(background_image, [combined_line], (255, 255, 255))

        return cv2.cvtColor(background_image, cv2.COLOR_RGB2GRAY)
        # return cv2.polylines(input_image, [combined_line], False, (0,255,255), thickness=10)

        # pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
        # pts = pts.reshape((-1,1,2))
        # return cv2.polylines(input_image, [pts], False, (0,255,255))

    def __calculate_known(self, expected_corner):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "warped")
        # It's now much easier to find line pixels!
        nonzero = self.warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        self.margin = 100
        self.left_lane_inds = ((self.nonzerox > (self.left_fit[0] * (self.nonzeroy ** 2) + self.left_fit[1] * self.nonzeroy +
                                       self.left_fit[2] - self.margin)) & (self.nonzerox < (self.left_fit[0] * (self.nonzeroy ** 2) +
                                                                             self.left_fit[1] * self.nonzeroy + self.left_fit[
                                                                                 2] + self.margin)))

        self.right_lane_inds = ((self.nonzerox > (self.right_fit[0] * (self.nonzeroy ** 2) + self.right_fit[1] * self.nonzeroy +
                                        self.right_fit[2] - self.margin)) & (self.nonzerox < (self.right_fit[0] * (self.nonzeroy ** 2) +
                                                                               self.right_fit[1] * self.nonzeroy + self.right_fit[
                                                                                   2] + self.margin)))

        # Again, extract left and right line pixel positions
        leftx = self.nonzerox[self.left_lane_inds]
        lefty = self.nonzeroy[self.left_lane_inds]
        rightx = self.nonzerox[self.right_lane_inds]
        righty = self.nonzeroy[self.right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        self.ploty = np.linspace(0, self.warped.shape[0] - 1, self.warped.shape[0])
        self.left_fitx = self.left_fit[0] * self.ploty ** 2 + self.left_fit[1] * self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0] * self.ploty ** 2 + self.right_fit[1] * self.ploty + self.right_fit[2]

    def __plot_known(self, ax, title):
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((self.warped, self.warped, self.warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx - self.margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx + self.margin,
                                                                        self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx - self.margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx + self.margin,
                                                                         self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        ax.imshow(result)
        ax.plot(self.left_fitx, self.ploty, color='yellow')
        ax.plot(self.right_fitx, self.ploty, color='yellow')
        ax.set_xlim(0, 1280)
        ax.set_ylim(720, 0)

    def measure_curvation(self):
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty)
        self.left_curverad = ((1 + (2 * self.left_fit[0] * y_eval + self.left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * self.left_fit[0])
        self.right_curverad = ((1 + (2 * self.right_fit[0] * y_eval + self.right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * self.right_fit[0])
        # print(self.left_curverad, self.right_curverad)
        # Example values: 1926.74 1908.48
        self.curv_diff = abs(self.left_curverad - self.right_curverad)
        # if abs(self.left_curverad - self.right_curverad) < self.curv_diff and \
        #         (self.left_curverad >= 0 and self.right_curverad >= 0 or self.left_curverad <= 0 and self.right_curverad <= 0):
        #     print('new curf diff {:6.0f} (new l - {:6.0f}, r - {:6.0f}'.format(self.curv_diff, self.left_curverad, self.right_curverad))
        # return True, self.left_curverad, self.right_curverad, self.curv_diff

        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.ploty * self.ym_per_pix, self.left_fitx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty * self.ym_per_pix, self.right_fitx * self.xm_per_pix, 2)
        # # Calculate the new radii of curvature
        self.left_curverad_m = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        self.right_curverad_m = ((1 + (
                    2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # # Now our radius of curvature is in meters
        return self.left_curverad_m, self.right_curverad_m

    def measure_center_difference(self):
        """
        :return: meters from center, pos is right, neg is left
        """
        left_lane_center_px = self.left_fitx[-1]
        right_lane_center_px = self.right_fitx[-1]
        
        current_center = (left_lane_center_px + right_lane_center_px) / 2

        meters_from_center = (current_center - 640 ) * self.xm_per_pix
        if meters_from_center > 0:
            txt = 'right'
        else:
            txt = 'left'

        return meters_from_center, txt

    def reset_curv_diff(self):
        self.curv_diff = sys.float_info.max