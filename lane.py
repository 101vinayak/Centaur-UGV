import cv2
import time
import numpy

ROI_START_Y = 150
FPS = 60
cap = cv2.VideoCapture('track_opencv.avi')
kernel = numpy.ones((5,5), numpy.uint8)

while cap.isOpened():
	ret, raw_frame = cap.read()
	if not ret:
		break
		
	frame = raw_frame[ROI_START_Y:]
	
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower_white = numpy.array([0,0,190])
	upper_white = numpy.array([255,20,255])
	
	mask = cv2.inRange(hsv, lower_white, upper_white)
	
	median = cv2.medianBlur(mask,5)
	
	opening = cv2.morphologyEx(median,cv2.MORPH_OPEN, kernel)
	dilation = cv2.dilate(opening, kernel, iterations=1)
	
	lines = cv2.HoughLinesP(dilation,1,numpy.pi/180,100,minLineLength=30,maxLineGap=150)
	
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
