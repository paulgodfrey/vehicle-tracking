import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import glob
import time

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.utils import shuffle

from skimage.feature import hog
from scipy.ndimage.measurements import label

import matplotlib.image as mpimg
import pickle
import imageio

from moviepy.editor import VideoFileClip
from IPython.display import HTML

import warnings
warnings.filterwarnings("ignore")

# Set default sobel thresholds
sobel_thresh_min = 10
sobel_thresh_max = 100

# Set placeholders for video labels
radius_label = ''
offset_label = ''

boxes = []
vehicles_detected = []
vehicles_centroid_last = []
scale = 1
current_frame = 0
debug = 0
offset_from_center_of_lane = 0

# Define a class to receive the characteristics of each lane line detected
class CameraCalibration():
    def __init__(self):
        self.ret = []
        # 3x3 floating-point camera matrix
        self.mtx = []
        # vector of distortion coefficients
        self.dist = []
        # vector of rotation vectors (see Rodrigues() ) estimated for each pattern view
        self.rvecs = []
        # vector of translation vectors estimated for each pattern view
        self.tvecs = []

# Define a class to receive the characteristics of each lane line detected
class Lane():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = []
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line

camera_cal = CameraCalibration()
left_lane = Lane()
right_lane = Lane()

# Calibrate camera distortion matrix given source images
def calibrate_camera(source_folder):
    global camera_cal

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)

    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(source_folder + '/cal*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Test undistortion on an image
    img = cv2.imread(source_folder + '/test_calibration.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    camera_cal.ret, camera_cal.mtx, camera_cal.dist, camera_cal.rvecs, camera_cal.tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)

        feature_image = convert_color(image, conv=color_space)

        spatial_features = bin_spatial(feature_image, size=spatial_size)
        hist_features = color_hist(feature_image, nbins=hist_bins)
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        features.append(np.hstack((spatial_features, hist_features, hog_features)))

    # Return list of feature vectors
    return features

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):

    feature_image = convert_color(img, conv=color_space)

    spatial_features = bin_spatial(feature_image, size=spatial_size)
    hist_features = color_hist(feature_image, nbins=hist_bins)
    hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)

    return np.hstack((spatial_features, hist_features, hog_features))

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)

    # Initialize a list to append window positions to
    window_list = []

    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def convert_color(img, conv):
    if conv == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def convert_labels_to_rectangles(labels):
    boxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        #cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        boxes.append((bbox[0], bbox[1]))
    # Return the image
    return boxes

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = hsl_convert(img)

    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 100)):
    # Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = hsl_convert(img)

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = hsl_convert(img)

    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_conversion(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Convert to single channel of HSL
def hsl_convert(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,1]

    return h_channel

def draw_gradient_thresholds(image):
    # Pull x and y gradients using sobel
    grad_bin_x = abs_sobel_thresh(image, orient='x', thresh_min=sobel_thresh_min, thresh_max=sobel_thresh_max)
    grad_bin_y = abs_sobel_thresh(image, orient='y', thresh_min=sobel_thresh_min, thresh_max=sobel_thresh_max)

    # Magnitude threshold
    mag_bin = mag_thresh(image, sobel_kernel=9, mag_thresh=(60,200))

    # Directional threshold
    dir_bin = dir_threshold(image, sobel_kernel=17, thresh=(1.7,2.0))

    # Combined gradient thresholds
    combined_bin = np.zeros_like(dir_bin)
    combined_bin[((grad_bin_x == 1) & (grad_bin_y == 1)) | ((mag_bin == 1) | (dir_bin == 1))] = 1

    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Threshold color channel
    s_thresh_min = 160
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Combine the two binary thresholds
    output_binary = np.zeros_like(combined_bin)
    output_binary[(s_binary == 1) | (combined_bin == 1)] = 1

    return output_binary

def draw_perspective_transform(source, destination, image):
    # Transform the image to birds eye view
    img_size = image.shape[0:2]
    img_size = img_size[::-1]

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(source, destination)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(image, M, img_size)

    return M, warped

def grid_search_for_lane(image):
    global left_lane
    global right_lane

    # Take a histogram of the bottom half of the image
    histogram = np.sum(image[image.shape[0]//2:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(image.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # warped_stream = warped

    left_lane.detected = 0
    right_lane.detected = 0

    if(left_lane.detected & right_lane.detected):
        left_fit = left_lane.current_fit
        right_fit = right_lane.current_fit

        # Search inside previous lane line area in new image for curviture
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin = 100

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
    else:
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window+1)*window_height
            win_y_high = image.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        left_lane.detected = 1
        right_lane.detected = 1

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def find_trailing_average_fit():
    global left_lane
    global right_lane

    sample_size = 10

    left_recent = left_lane.recent_xfitted[-(sample_size):]
    left_trailing_average = np.zeros(shape=(720))

    right_recent = right_lane.recent_xfitted[-(sample_size):]
    right_trailing_average = np.zeros(shape=(720))

    for i in range(len(left_recent)):
        left_trailing_average += np.array(left_recent[i])
        right_trailing_average += np.array(right_recent[i])

    left_trailing_average = left_trailing_average / sample_size
    right_trailing_average = right_trailing_average / sample_size

    return left_trailing_average, right_trailing_average

def find_lanes_in_frame(img):
    global current_frame
    global radius_label
    global offset_label
    global left_lane
    global right_lane
    global offset_from_center_of_lane

    # Undistort source image
    image = cv2.undistort(img, camera_cal.mtx, camera_cal.dist, None, camera_cal.mtx)

    # Find gradient thresholds in image to help extran lane lines
    gradient_thresholds = draw_gradient_thresholds(image)

    # Transform the image to birds eye view
    img_size = image.shape[0:2]
    img_size = img_size[::-1]

    # Define ponts on source image to transform
    top_left = [565, 450]
    top_right = [715, 450]
    bottom_right = [img_size[0] - 50, img_size[1] - 50]
    bottom_left = [50, img_size[1] - 50]

    # For source points I'm grabbing the outer four detected corners
    src = np.float32([top_left,
                      top_right,
                      bottom_right,
                      bottom_left])

    # Pick edge of image frame for destination points
    dst = np.float32([[0, 0], # top left
                     [img_size[0], 0], # top right
                     [img_size[0], img_size[1]], # bottom right
                     [0, img_size[1]]]) # bottom left

    # Transform the binary image to an overhead view and save the matrix coordinates
    Minv, warped = draw_perspective_transform(src, dst, gradient_thresholds)

    # Isolate lane lines by doing a grid search
    leftx, lefty, rightx, righty = grid_search_for_lane(warped)

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Store lane fits in global class
    left_lane.current_fit = left_fit
    right_lane.current_fit = right_fit

    # Generate x and y values for plotting
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Append latest fits to global class
    left_lane.recent_xfitted.append(left_fitx)
    right_lane.recent_xfitted.append(right_fitx)

    # If we have more than 10 frames of data use trailing average
    if(current_frame > 10):
        left_fitx, right_fitx = find_trailing_average_fit()

    # Define y-value where we want to evaluate radius of curvature
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_lane.radius_of_curvature = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_lane.radius_of_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Calculate vehicles distances from center of lane
    lane_center = ((right_fitx[-1] - left_fitx[-1])*0.5) + left_fitx[-1]
    camera_center = img_size[0]*0.5
    offset_from_center_of_lane = (camera_center - lane_center)*xm_per_pix*-1

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Invert the transform matrix to map the birds eye view back to the road
    invert_transform = cv2.invert(Minv)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, invert_transform[1], (image.shape[1], image.shape[0]))

    return newwarp

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    # Create an empty list to receive positive detection windows
    on_windows = []

    count = 0

    # Iterate over all windows in the list
    for window in windows:

        count += 1

        # Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        # Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # Predict using your classifier
        prediction = clf.predict(test_features)

        # If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)

    #8) Return windows for positive detections
    return on_windows

def grid_search(image, y_start_stop, x_start_stop, window_size, window_overlap):
    windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                        xy_window=window_size, xy_overlap=window_overlap)

    hot_spots = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    return hot_spots

def find_vehicles_in_frame(img):
    global current_frame
    global vehicles_detected
    global vehicles_centroid_last
    global radius_label
    global offset_label

    # Undistort source image
    image = cv2.undistort(img, camera_cal.mtx, camera_cal.dist, None, camera_cal.mtx)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    draw_img = np.copy(image)

    # store multiple grid searches in a single list
    hot_spots_coll = []

    # search for vehicles across various window sizes and xy coords
    hot_spots = grid_search(image, [350, 500], [600, 1280], (75, 75), (0.7, 0.7))
    hot_spots_coll += hot_spots

    hot_spots = grid_search(image, [375, 650], [600, 1280], (90, 90), (0.7, 0.7))
    hot_spots_coll += hot_spots

    hot_spots = grid_search(image, [400, 650], [600, 1280], (120, 120), (0.7, 0.7))
    hot_spots_coll += hot_spots

    hot_spots = grid_search(image, [500, 650], [600, 1280], (140, 140), (0.7, 0.7))
    hot_spots_coll += hot_spots

    hot_spots = hot_spots_coll

    # plot the raw vehicle detection to debug prior to grouping
    if(debug):
        draw_img_raw = draw_boxes(image, hot_spots, color=(0, 0, 255), thick=2)
        plt.imshow(draw_img_raw)
        plt.show()

    # Add heat to each box in box list
    heat = add_heat(heat, hot_spots)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1.5)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    # Get a list of rectangles to plot
    vehicles = convert_labels_to_rectangles(labels)

    # Store vehicle coordinates in a "buffer"
    for i in vehicles:
        vehicles_detected.append([i[0][0], i[0][1], i[1][0], i[1][1]])

    print('vehicles detected', vehicles_detected)

    # find lane lines in image
    lane_box = find_lanes_in_frame(image)

    # Allow buffer to build before grouping rectangles
    if(len(vehicles_detected) > 6):
        # Group rectangles that overlap
        vehicles_centroid, weights = cv2.groupRectangles(np.array(vehicles_detected).tolist(), groupThreshold=4)

        print('centroids detected:', len(vehicles_centroid))

        # Use previous coordinates in num changes to prevent single frame false positives
        if(len(vehicles_centroid) == len(vehicles_centroid_last)):
            print('used newly extracted centroids')
            for i in vehicles_centroid:
                print(i)
                cv2.rectangle(image, (i[0], i[1]), (i[2], i[3]), (0,0,255), 6)
        else:
            print('centroid len changed - used previous frame data')
            for i in vehicles_centroid_last:
                cv2.rectangle(image, (i[0], i[1]), (i[2], i[3]), (0,0,255), 6)

        # Update last used centroid list
        vehicles_centroid_last = vehicles_centroid

        #Trim stale data out of collection
        vehicles_detected = vehicles_detected[-20:]
    else:
        if(len(vehicles_centroid_last) > 0):
            print('no new data - used previous frame data')
            for i in vehicles_centroid_last:
                cv2.rectangle(image, (i[0], i[1]), (i[2], i[3]), (0,0,255), 6)

    image = cv2.addWeighted(image, 1, lane_box, 0.3, 0)

    # Update curvature reading every 6 frames
    if(current_frame % 6 == 0):
        radius_label = 'Radius of curvature is ' + str(int(left_lane.radius_of_curvature)) + 'm'
        offset_label = 'Vehicle is ' + str(round(offset_from_center_of_lane, 2)) + 'm left of center'

    font = cv2.FONT_HERSHEY_SIMPLEX

    result = cv2.putText(image, radius_label, (20, 40), font, 1, (255,255,255), 2)
    result = cv2.putText(image, offset_label, (20, 80), font, 1, (255,255,255), 2)

    current_frame += 1

    return image

# Read in cars and notcars
images = glob.glob('./training_data/all/*.jpeg')

cars = []
notcars = []

for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)

color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

spatial_size = (32, 32) # Spatial binning dimensions
spatial_feat = True # Spatial features on or off

hist_bins = 32 # Number of histogram bins
hist_feat = True # Histogram features on or off

orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
hog_feat = True # HOG features on or off

car_features = extract_features(cars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

notcar_features = extract_features(notcars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)

# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)

scaled_X, y = shuffle(scaled_X, y, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC
svc = LinearSVC()

# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()

print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

# Flags to either process video stream or single image
process_video = 1
process_image = 0
debug = 0

# Perform camera calibration so we can undistort input
calibrate_camera('camera_cal')

# Process video
if(process_video):
    white_output = 'vehicle_detection_output.mp4'
    clip1 = VideoFileClip("test_video_2.mp4")
    white_clip = clip1.fl_image(find_vehicles_in_frame)
    white_clip.write_videofile(white_output, audio=False)


# Process single image
if(process_image):
    image = mpimg.imread('./test_images/bbox-example-image.jpg')
    result = find_vehicles_in_frame(image)
    plt.imshow(result)
    plt.show()
