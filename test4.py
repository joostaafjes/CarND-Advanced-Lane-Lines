import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./test_images/test1.jpg')

# Create a black image
# img = np.zeros((512,512,3), np.uint8)

cv2.line(img, (10, 10), (100, 200), 100, 10)

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img, [pts], False, (0,255,255))

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Hello World!',(10,500), font, 1,(255,255,255),2)

cv2.imwrite('./output_images/test1-modified.jpg', img)


from pylab import figure, axes, pie, title, show

# Make a square figure and axes
fig = figure(1, figsize=(6, 6))
ax = axes([0.1, 0.1, 0.8, 0.8])

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
fracs = [15, 30, 45, 10]

explode = (0, 0.05, 0, 0)
pie(fracs, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
title('Raining Hogs and Dogs', bbox={'facecolor': '0.8', 'pad': 5})

# show()  # Actually, don't show, just save to foo.png

plt.savefig('./output_images/pychart2.jpg')

plt.close(fig)