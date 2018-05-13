# -*- coding: utf-8 -*-
'''
this is for setting the camera
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
cam= cv2.VideoCapture('/home/shorav/lego_proj/video/output1.avi')
cap1 = cv2.VideoCapture('/home/shorav/lego_proj/video/output1.avi')
#cap2 = cv2.VideoCapture(0)


while(True):
    # Capture frame-by-frame
    ret, frame = cam.read()
    frame[100:105,:,1]=255
    frame[400:405,:,1]=255
    frame[:,100:105,1]=255
    frame[:,500:505,1]=255
    
    
    
    ret2,frame2=cap1.read()
    frame2[100:105,:,1]=255
    frame2[400:405,:,1]=255
    frame2[:,100:105,1]=255
    frame2[:,500:505,1]=255
    
    
#    ret3,frame3=cap2.read()
#    frame3[100:105,:,1]=255
#    frame3[400:405,:,1]=255
#    frame3[:,100:105,1]=255
#    frame3[:,500:505,1]=255
#    
    #numpy_concat=np.concatenate((frame,frame2),axis=2)
    #numpy_concat2=np.concatenate((numpy_concat,frame),axis=1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    f, axarr = plt.subplots(2,1)
    axarr[0,0].imshow(gray)
    axarr[0,1].imshow(gray)
#    axarr[1,0].imshow(frame)
#    axarr[1,1].imshow(frame)

#    # Display the resulting frame
#    cv2.imshow('frame',frame)
#    
#    cv2.imshow('Top View', frame2)
#    
#    cv2.imshow('Front View', frame3)
    #cv2.imshow('numpy',numpy_concat)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

## When everything done, release the capture
cap.release()
cv2.destroyAllWindows()