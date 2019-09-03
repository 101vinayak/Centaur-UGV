# g,h,k
# mark point at distance theta fixed from lane 
# mark centre of the image as camera position 
# mark distance between the point and the camera position


# TODO : correct thresholds for the frame2 

import numpy as np 
import cv2
import glob
import matplotlib.pyplot as plt 
import pickle

import numpy as np
import cv2
from mark_frame import get_marked_frame as gmf 

from time import time 



# VIDEO / CAMERA :
# vidname = 'g.avi'
# vidname2 = 'g.avi' #using g.avi for cam2 as well with flipped images.
# cap = cv2.VideoCapture('test_vids/'+vidname)
# cap2 = cv2.VideoCapture('test_vids/'+vidname2)

cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)


# cv2.flip( img, 0 )

# CHECK VERSION :
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



start = time()

iframecnt = 0

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

frame_width2 = int(cap2.get(3))
frame_height2 = int(cap2.get(4))

import imutils

out = cv2.VideoWriter('combined'+str(time())+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps/2, (frame_width+frame_width2,frame_height))

print frame_width

while(cap.isOpened()):
    ret,frame = cap.read()
    ret2,frame2 = cap2.read()


    # cv2.imshow('camera1before',frame)
    # cv2.imshow('camera2before',frame2)


    frame = imutils.rotate(frame, 90)
    frame2 = imutils.rotate(frame2, 270)


    # rows,cols,_ = frame.shape
    # M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    # dst = cv2.warpAffine(frame,M,(cols,rows))

    # rows2,cols2,_ = frame2.shape
    # M2 = cv2.getRotationMatrix2D((cols2/2,rows2/2),90,1)
    # dst2 = cv2.warpAffine(frame2,M,(cols2,rows2))

    # cv2.imshow('camera1after',frame)
    # cv2.imshow('camera2after',frame2)

    # only for g.avi , flip the frame for the purpose of simulation.
    # frame2 = cv2.flip(frame2,1)
    
    if iframecnt < 100 :
        iframecnt+=1
        continue

    iframecnt+=1 
    if iframecnt %100 == 0:
        # print '[mark_frame]: iframecnt:',iframecnt
        pass
    if iframecnt%2==0:
        continue

    # print ret

    marked_image_cam1=gmf(ret=ret2,frame=frame2,framewidth=frame_width2,
                        frameheight=frame_height2,camera='r',
                        videowrite=False)

    marked_image_cam2=gmf(ret=ret,frame=frame,framewidth=frame_width,
                    frameheight=frame_height,camera='l',
                    videowrite=False)
    # print marked_image.shape
    # cv2.imshow('marked_image',marked_image)
    # cv2.imshow('mac')



    h1, w1 = marked_image_cam1.shape[:2]
    h2, w2 = marked_image_cam2.shape[:2]

    #create empty matrix
    vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

    #combine 2 images
    vis[:h1, :w1,:3] = marked_image_cam1
    vis[:h2, w1:w1+w2,:3] = marked_image_cam2

    cv2.line(vis,(vis.shape[1]//2,0),(vis.shape[1]//2,vis.shape[0]),(255,0,0),3)
    cv2.imshow('Camera',vis)
    out.write(vis)
    cv2.waitKey(1)

end = time()

cap.release()
# out.release()
cv2.destroyAllWindows()


timereq= end-start
print 'time_req =',timereq
