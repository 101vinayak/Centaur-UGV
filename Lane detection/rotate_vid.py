import cv2
import imutils

filename = "one1522932002.05.avi"
cap = cv2.VideoCapture(filename)
out = cv2.VideoWriter('rotated_'+filename,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(cap.get(3)),int(cap.get(4))))

while cap.isOpened():
	ret, frame = cap.read()
	frame = imutils.rotate(frame, 90)
	out.write(frame)