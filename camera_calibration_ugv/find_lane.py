import cv2
import time
import numpy as np
import cv2
import glob
import pickle as pkl 


from caliberation_class import caliberation 

with open('caliberation_vals.pkl','rb') as infile :
	calib_vals = pkl.load(infile)

roi = calib_vals.roi
h = calib_vals.h
w = calib_vals.w
newcameramtx = calib_vals.newcameramtx
mtx = calib_vals.mtx
dist = calib_vals.dist



ROI_START_Y = 150
FPS = 60
cap = cv2.VideoCapture('track_opencv.avi')
# cap = cv2.VideoCapture(1)
kernel = np.ones((5,5), np.uint8)

while cap.isOpened():
	# break
	ret, raw_frame = cap.read()
	if not ret:
		break

	frame = cv2.resize(raw_frame , (640,480))
	# frame = frame[ROI_START_Y:]

	dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
	x, y, w, h = roi


	dst = dst[y:y+h, x:x+w]
	# cv2.imshow('framee',dst)
	frame = dst
	
	hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
	lower_white = np.array([0,0,190])
	upper_white = np.array([255,20,255])
	
	mask = cv2.inRange(hsv, lower_white, upper_white)
	
	median = cv2.medianBlur(mask,5)
	
	opening = cv2.morphologyEx(median,cv2.MORPH_OPEN, kernel)
	dilation = cv2.dilate(opening, kernel, iterations=1)
	
	lines = cv2.HoughLinesP(dilation,1,np.pi/180,100,minLineLength=30,maxLineGap=150)
	
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line[0]
			cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
	cv2.imshow('frame', frame)
	#cv2.imshow('result', dilation)
	
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
	time.sleep(1.0/FPS)	
	
cap.release()
cv2.destroyAllWindows()
