import cv2 
import numpy as np

cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)
from time import time


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver)  < 3 :
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    fps2 = cap2.get(cv2.cv.CV_CAP_PROP_FPS)
    print "Frames per second Camera1: {0}".format(fps)
    print "Frames per second Camear2: {0}".format(fps2)
else :
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    print "Frames per second Camera1: {0}".format(fps)
    print "Frames per second Camera2: {0}".format(fps2)



# IF OUTPUT 
# out = cv2.VideoWriter('out_test_vids/'+vidname,cv2.VideoWriter_fourcc('M','J','P','G'), fps/2, (frame_width,frame_height))

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

frame_width2 = int(cap2.get(3))
frame_height2 = int(cap2.get(4))

out = cv2.VideoWriter('one'+str(time())+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps/2, (frame_width,frame_height))
out2 = cv2.VideoWriter('two'+str(time())+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps2/2, (frame_width2,frame_heigh2))
# out.write(image)
flag = False

while True :
	ret,frame = cap.read()
	ret,frame2 = cap2.read()

	if flag is False :
		print frame.shape 
		print frame2.shape

	flag = True 


	out.write(frame)
	out2.write(frame2)
	cv2.imshow('camera1',frame)
	cv2.imshow('camera2',frame2)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cap2.release()
out.release()
out2.release()
cv2.destroyAllWindows()
