# import the necessary packages
import numpy as np
import cv2
 
def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	val = ((2.0, 117.0), (4.0, 234.0), -0.0)
	try:
		kernel = np.ones((5,5),np.uint8)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (5, 5), 0)
		cv2.imshow('gray',gray)
		edged = cv2.Canny(gray, 50, 65)
		# edged = cv2.dilate(edged,kernel,iterations = 3)

		cv2.imshow('edged',edged)
	 
		# find the contours in the edged image and keep the largest one;
		# we'll assume that this is our piece of paper in the image
		(_,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		# print cnts

		# new_c = []
		# for cc in cnts :
		# 	ac = cv2.contourArea(cc) 
		# 	if ac>40 and ac < 55 :
		# 		new_c.append(cc)


		c = max(cnts, key = cv2.contourArea)
		area = cv2.contourArea(c)
		print area
	 	return cv2.minAreaRect(c)
	except :
		return val
	# compute the bounding box of the of the paper region and return it
	# return cv2.minAreaRect(c)
 
def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth
 
# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 12
# initialize the known object width, which in this case, the piece of
# paper is 11 inches wide
KNOWN_WIDTH = 2
 
 
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread('img.png')
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

# draw a bounding box around the image and display it
box = np.int0(cv2.boxPoints(marker))
cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
cv2.putText(image, "%.2fft" % (inches / 12),
	(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
	2.0, (0, 255, 0), 3)
cv2.imshow("given_img", image)
# loop over the images
# for imagePath in IMAGE_PATHS:

cap = cv2.VideoCapture(1)
while True :
	# load the image, find the marker in the image, then compute the
	# distance to the marker from the camera
	# image = cv2.imread(imagePath)

	ret, frame = cap.read()
	# cv2.imshow('original',frame)
	marker = find_marker(frame)
	inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
 
	# draw a bounding box around the image and display it
	box = np.int0(cv2.boxPoints(marker))
	cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
	cv2.putText(frame, "%.2fft" % (inches / 12),
		(frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		2.0, (0, 255, 0), 3)
	cv2.imshow("frame", frame)
	cv2.waitKey(300)


cap.release()
cv2.destroyAllWindows()