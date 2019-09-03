
import numpy as np 
import cv2
import glob
import matplotlib.pyplot as plt 
import pickle
import imutils

# ii = raw_input()


import numpy as np
import cv2




def get_marked_frame(ret,frame,framewidth,frameheight,camera='l',videowrite=False):

    if ret is False :
        return
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5,5), np.uint8)
    # imgname = 'b.png'
    # imgname = ii + '.png'
    # image = cv2.imread(imgname)
    #cv2.imshow('frame',image)
    # imagea = cv2.imread(imgname)
    image = frame 
    imagea = frame 
    _N_ = 10

    img_size = (1366, 768)


    def pipeline (image):

        img = np.copy(image)
        img = cv2.GaussianBlur(img,(5,5),0)

        hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS).astype(np.float)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)

        h_channel_hsv = hls[:,:,0]
        s_channel_hsv = hls[:,:,1]
        v_channel_hsv = hls[:,:,2]

        lower_white = np.array([0,0,190])
        upper_white = np.array([255,20,255])

        maskw = cv2.inRange(hsv, lower_white, upper_white)
        # cv2.imshow('mask',maskw)

        # bitwise and of mask and image
        andimg = cv2.bitwise_and(image,image,mask = maskw).astype(np.float)
        # cv2.imshow('and image',andimg)

        median = cv2.medianBlur(maskw,5)
        # cv2.imshow('median',median)

        #opening = cv2.morphologyEx(median,cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(maskw, kernel, iterations=1)
        #cv2.imshow('opening',opening)
        # cv2.imshow('closing',dilation)

        return dilation
        #img_func = l_channel
        #ksize = 15
        #mag_binary = mag_thresh(img_func,sobel_kernel = ksize, mag_thresh=(30,255), switch_gray = False)
        #dir_binary = dir_threshold(img_func,sobel_kernel = ksize, 
    class Line():
        def __init__(self):
            # for how many iterations was the lane not dectected
            self.undetected_ct = 0  
            # x values of the last n fits of the line
            self.recent_xfitted = [] 
            #average x values of the fitted line over the last n iterations
            self.bestx = None     
            #polynomial coefficients of the last n iterations
            self.recent_best_fit = [] 
            #polynomial coefficients averaged over the last n iterations
            self.best_fit = None  
            #polynomial coefficients for the most recent fit
            self.current_fit = [np.array([False])]  
            #radius of curvature of the line in meters
            self.radius_of_curvature = None 
            #distance in meters of vehicle center from the line
            self.line_base_pos = None 
            #difference in fit coefficients between last and new fits
            self.diffs = np.array([0,0,0], dtype='float') 
            #x values for detected line pixels
            self.allx = None  
            #y values for detected line pixels
            self.framect = 0  
            #y values for detected line pixels

    N_ = 10 #average the values for the last _N_ lines
    _MIN_PIXELS_ = 100 # enough pixels to identify a line
    ym_per_pix = 0.0166 # meters per pixel in y dimension
    xm_per_pix = 3.7/1000 # meters per pixel in x dimension
    _MAX_RADIUS_RATIO_ = 10
            
    def identify_lane_from_existing(img, lane_record = None):
        
        if lane_record is None:
            return None
        
        else:
        
            # get the points needed: 
            side_lane = np.copy(img)

            yvals = np.arange(10)*img_size[0]/10 + 72/2
            side_fitx = lane_record.best_fit[0]*yvals**2 + lane_record.best_fit[1]*yvals + lane_record.best_fit[2]

            existing_mask = np.ones(shape = (img.shape[0], img.shape[1]))
            for i in range(10):
                existing_mask[int(i*side_lane.shape[0]/10):int((i+1)*side_lane.shape[0]/10),:int(side_fitx[i]) - 25] = 0
                existing_mask[int(i*side_lane.shape[0]/10):int((i+1)*side_lane.shape[0]/10),int(side_fitx[i]) + 25:] = 0

            for i in range(10):
                side_lane[int(i*side_lane.shape[0]/10):int((i+1)*side_lane.shape[0]/10),:int(side_fitx[i]) - 25] = 0
                side_lane[int(i*side_lane.shape[0]/10):int((i+1)*side_lane.shape[0]/10),int(side_fitx[i]) + 25:] = 0

            if DEBUG:

                f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,4))
                # export_binary(binary = birdeye_img, dst = 'output_images/identify_lane_v2_birdeyeimage.jpg')
                ax1.imshow(img, cmap = 'gray')
                ax2.set_title("Image")
                # export_binary(binary = existing_mask, dst = 'output_images/identify_lane_v2_existing_mask.jpg')
                ax2.imshow(existing_mask, cmap = 'gray')
                ax2.set_title("Existing Mask")
                # export_binary(binary = side_lane, dst = 'output_images/identify_lane_v2_identified_lane.jpg')
                ax3.imshow(side_lane, cmap = 'gray')
                ax3.set_title("Identified Lane")
                plt.show()

            return side_lane


    def identify_lane(img, lane = 'left', lane_record = None):
        # DEBUG = True
        is_hist = True
        
        # if there are enough values in the lane record and the last lane was detected, look for a new line around the existing one.
        if (lane_record is not None and lane_record != None and len(lane_record.recent_xfitted) >= _N_ and lane_record.undetected_ct == 0):
            print '[mark_frame]: Existing'
            
            print('identify_lane: Using existing values')
            side_lane = identify_lane_from_existing(img, lane_record = lane_record)
            is_hist = False
        
        else:
            # print 'lane_record is none in identify_lane'
            if DEBUG:
                print('identify_lane: Running histogram search')
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,2))            
                
                # export_binary(binary = birdeye_img, dst = 'output_images/identify_lane_orgininal_mask.jpg')
                ax1.imshow(img, cmap = 'gray')
            
            img1= imutils.rotate(img, 90)
            
            histogram = np.mean(img[int(img.shape[0]/2):,:], axis=0)
            histogram1 =np.mean(img1[int(img1.shape[0]/2):,:], axis=0)

            if DEBUG:
                plt.plot(histogram)
                plt.show()

            center_point = int(img.shape[1]/2)

            histogram_side = np.copy(histogram)

            center_point1 = int(img1.shape[1]/2)

            histogram_side1 = np.copy(histogram1)
            
            argmax_histogram_side = np.argmax(histogram_side)
            argmax_histogram_side1 = np.argmax(histogram_side1)
           ####### we'll compare argmax from both axis and then use the one with larger val 
            if argmax_histogram_side1>argmax_histogram_side:
                argmax_histogram_side=argmax_histogram_side1
                img = img1
                
            side_min_lane_detected = argmax_histogram_side - 50 
            side_max_lane_detected = argmax_histogram_side + 50 
                  
            # PROGRESSIVELY IDENTIFY THE LINE
            img_copy = np.copy(img)
            side_ranges = np.ndarray(shape=(8,2), dtype=float)
            side_range_min = side_min_lane_detected
            side_range_max = side_max_lane_detected

            # redo a histogram within the range
            for i in range(8): 
                
                lower_part = img_copy[int((7-i)*img.shape[0]/8):int((7-i+1)*img.shape[0]/8),:]
                lower_part[:,:side_range_min] = 0
                lower_part[:,side_range_max:] = 0
                hist = np.mean(lower_part, axis = 0)
                avgpoint = np.argmax(hist)
                if(hist[avgpoint] > .1):
                    side_range_min = max(0,avgpoint - 50)
                    side_range_max = min(img_size[1], avgpoint + 50)
                else:
                    side_range_min = max(0,side_range_min - 50)
                    side_range_max = min(img_size[1],side_range_max + 50)
                side_ranges[7-i] = [side_range_min, side_range_max]
                
                if DEBUG:
                    plt.subplot(8,1,8-i)
                    plt.plot(hist)
                    plt.tick_params(axis='both', top='off', right='off', bottom='off', labeltop='off', labelright='off', labelbottom='off')

            if DEBUG:
                plt.show()

            # SELECT ONLY THE PIXELS IDENTIFIED ABOVE
            side_lane = np.copy(img)
            for i in range(8):
                side_lane[int(i*side_lane.shape[0]/8):int((i+1)*side_lane.shape[0]/8),:int(side_ranges[i][0])] = 0
                side_lane[int(i*side_lane.shape[0]/8):int((i+1)*side_lane.shape[0]/8),int(side_ranges[i][1]):] = 0
            
            
        # FIT LANE
        side_lane[side_lane > .5] = 1
        side_lane[side_lane < .5] = 0

        vals = np.argwhere(side_lane>.5)
        sidex = vals.T[1]
        sideyvals = vals.T[0]

        if len(sideyvals) >= _MIN_PIXELS_: # enough pixels to identify a line
            side_fit = np.polyfit(sideyvals, sidex, 2)
        else:
            side_fit = None
        
        # export_binary(binary = side_lane, dst = 'output_images/identify_lane_side_lane.jpg')
        
        return side_fit, sideyvals, sidex, is_hist


    # img = cv2.imread(imgname)
    img = frame 
    img_filter= pipeline(img)
    # cv2.imshow('filter',img_filter)
    DEBUG = False
    left_fit, leftyvals, leftx, is_hist = identify_lane(img_filter, lane = 'left')




    def get_curb(fit, sideyvals, sidex):
        if (fit is None or sideyvals is None or sidex is None):
            return None, None
        
        y_eval = img_size[0]
        #side_curverad_px = ((1 + (2*fit[0]*y_eval + fit[1])**2)**1.5) \
        #                             /np.absolute(2*fit[0])    
        side_line_position_px = fit[0]*y_eval**2 + fit[1]*y_eval + fit[2]
        

        y_eval = img_size[0]/2
        # print 'GetCurb: ',sideyvals,sidex,ym_per_pix,xm_per_pix 
        side_fit_cr = np.polyfit(sideyvals*ym_per_pix, sidex*xm_per_pix, 2)

        side_curverad_meters = ((1 + (2*side_fit_cr[0]*y_eval + side_fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*side_fit_cr[0])
        
        # Now our radius of curvature is in meters
        if DEBUG:
            print(int(side_curverad_meters), 'm')

        return side_curverad_meters, side_line_position_px

    def get_best_curb(lane_record):
        if (lane_record is None or lane_record.best_fit is None):
            return None
        
        y_eval = img_size[0]/2
        
        yvals = np.arange(11)*img_size[0]/10
        side_fitx = lane_record.best_fit[0]*yvals**2 + lane_record.best_fit[1]*yvals + lane_record.best_fit[2]

        y_eval = img_size[0]
        side_line_position_px = lane_record.best_fit[0]*y_eval**2 + lane_record.best_fit[1]*y_eval + lane_record.best_fit[2]
        
        y_eval = img_size[0]/2
        side_fit_cr = np.polyfit(yvals*ym_per_pix, side_fitx*xm_per_pix, 2) 
        side_curverad_meters = ((1 + (2*side_fit_cr[0]*y_eval + side_fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*side_fit_cr[0])
        
        # Now our radius of curvature is in meters
        if DEBUG:
            print(int(side_curverad_meters), 'm')

        return side_curverad_meters, side_line_position_px


    def get_car_position(side_line_position):
        car_position_px = abs(img_size[1]/2 - side_line_position)   #pixels between left line and the center of the car i.e. center of x-axis   
        car_position_meters = car_position_px * xm_per_pix    
        return car_position_meters

    def get_car_from_middle(left_lane):
        
        if (left_lane is None or left_lane.line_base_pos is None):
            return 0 
        
        center = (left_lane.line_base_pos)
        if DEBUG:
            print('left_lane.line_base_pos:', left_lane.line_base_pos)
        return center - left_lane.line_base_pos


    left_curverad_meters, left_line_position_px = get_curb(left_fit, leftyvals, leftx)
    #right_curverad_meters, right_line_position_px = get_curb(right_fit, rightyvals, rightx)

    # if left_line_position_px is None :
    #     print 'left_line_position_px is None'
    # try :
    #     print('Meters: left_curverad - right_curverad:', int(left_curverad_meters))
    #     print('Pixels: left_line_position_px - right_line_position_px:', int(left_line_position_px))
    # except :
    #     print('Meters: left_curverad - right_curverad:', left_curverad_meters )
    #     print('Pixels: left_line_position_px - right_line_position_px:', left_line_position_px )

    def update_lane_info(lane, side_fit, side_curverad_meters, side_line_position, sideyvals, sidex, ignore_frame_side = False): 
        # print 'IN UPDATE_LANE_INFO:'
        # print lane 
        # print side_fit
        # print side_curverad_meters
        # print side_line_position
        # print sideyvals
        # print sidex
        # print ignore_frame_side
        # print '-'*10


        if DEBUG:
            print('side_curverad_meters:', side_curverad_meters)
            print('side_fit:', side_fit)
        
        ignore_frame = ignore_frame_side
        
        # decide if we should keep the lane or not:
        old_fit = lane.current_fit
        
        if side_fit is None:
            ignore_frame = True
        
        if (side_fit is not None and len(lane.recent_xfitted) >= 4): 
            fit_difference = side_fit - old_fit
            fit_difference_norm = np.sqrt(np.sum((side_fit[0]-old_fit[0])**2))
            if DEBUG:
                print('fit_difference_norm:', fit_difference_norm)
                print('side_fit:', side_fit, 'old_fit:', old_fit)
            if (lane.undetected_ct < 20 and fit_difference_norm > .0005):
                ignore_frame = True

        if ignore_frame == False:
            # x values of the last n fits of the line 
            if (len(lane.recent_xfitted) > 10):
                _ = lane.recent_xfitted.pop()
            lane.recent_xfitted.insert(0, side_line_position)

            #average x values of the fitted line over the last n iterations
            lane.bestx = np.mean(lane.recent_xfitted)

            if (len(lane.recent_best_fit) > 10):
                _ = lane.recent_best_fit.pop()
            lane.recent_best_fit.insert(0, side_fit)

            #polynomial coefficients averaged over the last n iterations
            if lane.best_fit is None:
                lane.best_fit = side_fit     
            else:
                lane.best_fit = np.mean(lane.recent_best_fit, axis = 0)

            #polynomial coefficients for the most recent fit
            lane.current_fit = lane.best_fit 

            #radius of curvature of the line in some units
            lane.radius_of_curvature,  side_line_position_px = get_best_curb(lane) 

            #distance in meters of vehicle center from the line
            lane.line_base_pos = get_car_position(side_line_position_px) 
            
            #difference in fit coefficients between last and new fits
            lane.diffs = side_fit - old_fit 

            #x values for detected line pixels
            lane.allx = sidex  

            #y values for detected line pixels
            lane.ally = sideyvals
            
            # reset undetected_ct
            lane.undetected_ct = 0
        else:
            if DEBUG:
                print('NOT UPDATING')

            lane.undetected_ct += 1
        lane.framect += 1
        
        return ignore_frame



    record_lane = Line()
    _ = update_lane_info(record_lane, left_fit, left_curverad_meters, left_line_position_px, leftyvals, leftx, False)
    #_ = update_lane_info(record_right_lane, right_fit, right_curverad_meters, right_line_position_px, rightyvals, rightx, False)





    def draw_lanes_image(image,img, record_lane):

        if (record_lane is None ):
            if DEBUG:
                print('Cannot draw_lanes_image since one is None')
            return image
        
        yvals = np.arange(11)*img_size[0]/10
        # print record_lane
        # print 'record_lane.best_fit:',record_lane.best_fit
        # print yvals
        try :
            left_fitx = record_lane.best_fit[0]*yvals**2 + record_lane.best_fit[1]*yvals + record_lane.best_fit[2]

            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
            pts = np.hstack(pts_left)

            cv2.fillPoly(image, np.int_([pts]), (0,255, 0))
            # cv2.imshow('frame',image)
            # print np.int_([pts])
            theta = 200 
            # print image.shape
            ff= np.int_([pts])
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            for points in ff[0] :
                # print points
                cv2.circle(image,(points[0],points[1]), 10, (0,255,255), -1)
                cv2.putText(image,str(points),(points[0],points[1]), font, 1,(255,255,255),2,cv2.LINE_AA)

            # print ff

            pos =(np.int_([pts])[0][3] + np.int_([pts])[0][2])/2
            # print pos
            pos[0]  = pos[0] -theta
            cv2.circle(image,(pos[0],pos[1]), 10, (0,0,255), -1)

            frame_mid_horizontal = image.shape[1]/2
            cv2.line(image,(frame_mid_horizontal,0),(frame_mid_horizontal,image.shape[0]),(255,0,0),3)


            cv2.line(image,(pos[0],pos[1]),(frame_mid_horizontal,pos[1]),(255,0,0),3)

            if (pos[0] < frame_mid_horizontal) :
                cv2.putText(image,'Move Left',(20,50), font, 1,(255,255,255),2,cv2.LINE_AA)
            else :
                cv2.putText(image,'Move Right',(20,50), font, 1,(255,255,255),2,cv2.LINE_AA)

            # cv2.imshow('frame',image)
            # cv2.waitKey(1)
            # out.write(image)
            # plt.imshow(image)
            # plt.show()
            return image
        except :
            pass

        return image



    record_lane = Line()
    # print 'Record_LANE :', record_lane

    debug = False

    def full_pipeline(image):

            
        result_pipeline  = pipeline(image) # it will change color spaces etc.

        if DEBUG:
            plt.imshow(result_pipeline)
            plt.show()
        
        #right_fit, rightyvals, rightx, is_hist_right = identify_lane(result_pipeline, lane = 'right', lane_record = record_right_lane)
        # print 'in func record lane:',record_lane
        left_fit, leftyvals, leftx, is_hist_left = identify_lane(result_pipeline, lane = 'left', lane_record = record_lane)

        left_curverad_meters, left_line_position_px = get_curb(left_fit, leftyvals, leftx)


        ignore_frame_left_ratio = False
        ratio_left_right = None
        # print 'record_lane.best_fit in full pipeline:',record_lane.best_fit
        ignore_frame_left = update_lane_info(record_lane, left_fit, left_curverad_meters, left_line_position_px, leftyvals, leftx, ignore_frame_left_ratio)
        # print 'record_lane.best_fit in full pipeline after:',record_lane.best_fit
        #ignore_frame_right = update_lane_info(record_right_lane, right_fit, right_curverad_meters, right_line_position_px, rightyvals, rightx, ignore_frame_right_ratio)
        
        return draw_lanes_image(image, result_pipeline, record_lane = record_lane)


     
    return full_pipeline(imagea)
