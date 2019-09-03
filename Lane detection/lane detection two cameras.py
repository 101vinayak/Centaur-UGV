# g,h,k
# mark point at distance theta fixed from lane 
# mark centre of the image as camera position 
# mark distance between the point and the camera position


# TODO : correct thresholds for the frame2 

import numpy as np 
import cv2
import matplotlib.pyplot as plt

import numpy as np
import cv2
import imutils

from time import time 

from mark_frame import get_marked_frame as gmf 
from mark_frame import mark_lane

import navigate as nav


# VIDEO / CAMERA :
vidname = '2_rotated.avi'
# vidname2 = 'oi2.avi' #using g.avi for cam2 as well with flipped images.
cap = cv2.VideoCapture('cameravids/leftcam/'+vidname)
#cap2 = cv2.VideoCapture('cameravids/rightcam/'+vidname)

FRAMEBUFFER_LEN = 5

# cap = cv2.VideoCapture(1)
# cap2 = cv2.VideoCapture(2)


# cv2.flip( img, 0 )

# CHECK VERSION :
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver)  < 3 :
    fps = 1 #cap.get(cv2.cv.CV_CAP_PROP_FPS)
    fps2 = 1 #cap2.get(cv2.cv.CV_CAP_PROP_FPS)
    print "Frames per second Camera1: {0}".format(fps)
    print "Frames per second Camear2: {0}".format(fps2)
else :
    fps = 1 #cap.get(cv2.CAP_PROP_FPS)
    fps2 = 1 #cap2.get(cv2.CAP_PROP_FPS)
    print "Frames per second Camera1: {0}".format(fps)
    print "Frames per second Camera2: {0}".format(fps2)



# IF OUTPUT 
# out = cv2.VideoWriter('out_test_vids/'+vidname,cv2.VideoWriter_fourcc('M','J','P','G'), fps/2, (frame_width,frame_height))


start = time()

iframecnt = 0

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# frame_width2 = int(cap2.get(3))
# frame_height2 = int(cap2.get(4))


print frame_width,frame_height
# print frame_width2,frame_height2


# out = cv2.VideoWriter('combined'+str(time())+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps/2, (frame_width+frame_width2,frame_height))

# print frame_width

framecounter=0
THRESHOLD = 50 # for lane delta, in pixels

framebuffer = []
while(cap.isOpened()):

    showoriginal = False
    # change below line back to frame and frame2
    ret,frame = cap.read()
    #ret2,frame2 = cap2.read()

    # if showoriginal:

    #     originalframe = frame.copy()
    #     originalframe2 = frame2.copy()


    centre_points, cur_angle, cur_delta, isRotated = gmf(ret,frame,frame_width, frame_height)
    framebuffer.append((cur_angle, cur_delta))
    
    if framecounter % FRAMEBUFFER_LEN == 0:
        print 'framebuffer', framebuffer
        (angle, delta) = np.median(framebuffer, axis=0)
        print 'angle=',angle
        marked_image_cam1 = mark_lane(frame, centre_points, angle, delta, isRotated)
        cv2.imshow('Camera_left',marked_image_cam1)
        framebuffer = []

    framecounter += 1
    # cv2.imshow('camera1before',frame)
    # cv2.imshow('camera2before',frame2)


    ######frame = imutils.rotate(frame, 90)
    #########frame2 = imutils.rotate(frame2, 270)


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

    #marked_image_cam2=gmf(ret=ret,frame=frame2,framewidth=frame_width,
    #                frameheight=frame_height,camera='l',
    #                videowrite=False)
    # print marked_image.shape
    # cv2.imshow('marked_image',marked_image)
    # cv2.imshow('mac')



    h1, w1 = marked_image_cam1.shape[:2]
    #h2, w2 = marked_image_cam2.shape[:2]
    #create empty matrix
    vis = np.zeros((h1, w1,3), np.uint8)
    #vis_r = np.zeros((h2, w2,3), np.uint8)
    #combine 2 images
    vis[:h1, :w1,:3] = marked_image_cam1
    #vis_r[:h2, :w2,:3] = marked_image_cam2


    if showoriginal:
        # h1, w1 = frame.shape[:2]
        # # h2, w2 = frame2.shape[:2]
        # #create empty matrix
        # vis2 = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
        # #combine 2 images
        # vis2[:h1, :w1,:3] = originalframe
        # vis2[:h2, w1:w1+w2,:3] = originalframe2
        cv2.imshow('Original', frame)




    #cv2.line(vis,(vis.shape[1]//2,0),(vis.shape[1]//2,vis.shape[0]),(255,0,0),3)
    #cv2.imshow('Camera_right', vis_r)

    # if delta < THRESHOLD and angle is not None:
    #     nav.turn(np.pi/2 - angle)
    #     nav.move_forward(nav.SLOW)

    # if showoriginal:
    #     cv2.imshow('Original',cv2.resize(vis2,(vis2.shape[1]//2,vis2.shape[0]//2)))
    # out.write(vis)
    cv2.waitKey(1)

end = time()

cap.release()
# out.release()
cv2.destroyAllWindows()


timereq= end-start
print 'time_req =',timereq