#Author : Mudit Verma
#Purpose : UGV-Nav

import cv2
import numpy as np

img = cv2.imread('a.png')
print(img.shape)


def draw_line_on_image(img,points=None,mode='vertical',color=(255,255,255),thickness=5,debug=False):
	'''
	@img : image
	@mode: vertical/horizontal  #vertical
	@points: ((x1,y1),(x2,y2)) in normal x,y coordinate system.
	@color: (b,g,r) tuple	#(255,255,255)
	@thickness: int value for line thickness	#5
	@debug: True to show line using cv2.imshow	#False
	'''	

	height_from_bottom_horizontal = img.shape[0]//2
	dist_from_left_vertical = img.shape[1] //2

	if mode == 'horizontal':
	# horizontal line
		cv2.line(img,(0,height_from_bottom_horizontal) , (img.shape[1],height_from_bottom_horizontal) , color , thickness)
	elif mode == 'vertical':
	# vertical line
		cv2.line(img,(dist_from_left_vertical,0) , (dist_from_left_vertical,img.shape[0]) , color , thickness)	
	else:
		if points is not None :
			(x1,y1),(x2,y2) = points
			cv2.line(img,(y1,x1) , (y2,x2) , color , thickness)
		else :
			print ('ERROR: [linedrawing-draw_line_on_image] : invalid mode/points given.')
			return

	if debug:
		while True :
			cv2.imshow('Image',img)
			cv2.waitKey(1)
	return img



def point_on_image(img,color=(0,255,255),size=10,thickness=-1,point=None,position=None,debug=False):
	'''
	@img : Image
	@point : tuple (x,y)	#bottom center
	@debug : True to show image 	#False
	'''


	if point is None:
		# img.shape -> [height,width]
		pointheight = img.shape[0]
		pointwidth = img.shape[1]//2
		point = (pointwidth,pointheight)

	cv2.circle(img,point, size, color, thickness)

	if debug:
		while True :
			cv2.imshow('Image',img)
			cv2.waitKey(1)

	return img



def draw_point2point(img,point1,point2,p1color=(255,0,0),p2color=(0,255,0),linecolor=(255,255,255),debug=False,thickness=5):
	point_on_image(img,point=point1,color=p1color)
	point_on_image(img,point=point2,color=p2color)
	points=(point1,point2)
	draw_line_on_image(img=img,points=points,mode='other',color=(0,255,255),debug=debug)
	return img


import imutils

if __name__ == '__main__':
	# img2 = draw_line_on_image(img,mode='horizontal')
	p1=(500,500)
	p2=(100,100)
	pointheight = img.shape[0]
	pointwidth = img.shape[1]//2
	point = (pointwidth,pointheight)


	img2 = imutils.rotate(img,90)
	# img2 = draw_point2point(img=img,point1=p1,point2=p2,debug=True)
	#r_point = 
	img = point_on_image(img,point=point)
	while True : 
		cv2.imshow('im',img2)
		cv2.imshow('imm orig',img)
		cv2.waitKey(1)

	cv2.destroyAllWindows()
