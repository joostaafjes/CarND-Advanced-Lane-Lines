import pickle
import cv2
from matplotlib import pyplot as plt

# Test undistortion on an image
img = cv2.imread('./camera_cal/test_image.jpg')
# img = cv2.imread('./test_images/test1.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_size = (img.shape[1], img.shape[0])

# load pickle with mtx and dist
dist_pickle = pickle.load(open("./camera_cal/wide_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

dst = cv2.undistort(img, mtx, dist, None, mtx)
# cv2.imwrite('../examples/test_undist.jpg',dst)

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
