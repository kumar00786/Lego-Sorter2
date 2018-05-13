# -*- coding: utf-8 -*-
"""
This code covers following things:-
1. Doing the Prediction of top image for the background subtraction, edge detection, inception and poet model 
2. Doing the prediction for side image for the background subtration, edge detection , inception and poet model
"""
import numpy as np
import cv2,time,os
from colorama import *
#cam= cv2.VideoCapture('/home/shorav/lego_proj/video/output1.avi')
cam= cv2.VideoCapture(0)## this is for upper camera
cam2=cv2.VideoCapture(1)## this is for lower camera
height = cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
width = cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
height=int(height)
width=int(width)
######### Tensor Flow##########################################################
'''
to implement the tensor flow following things are needed:-
1. To store the path of the file 
2. Graph in the .pb format in the same folder
3. labels of all the category in the text file 
'''
import tensorflow as tf
### path for the top image edge detection#############################
#top_edge='/home/shorav/all_codes/edge_detection/top_view/top_edge.jpg'
### path for the top iamge inception model############################
top_inception='/home/shorav/all_codes/inception/top_view/frame1.jpg'
top_inception2='/home/shorav/all_codes/inception/top_view/'
### path for the top image poet model################################
#top_poet='/home/shorav/all_codes/poet/top_view/top_poet.jpg'
#### path for the top image background subtraction####################
#top_background='/home/shorav/all_codes/background_subtraction/top_view/top_background.jpg'
#
#
### path for the side image edge detection#############################
#side_edge='/home/shorav/all_codes/edge_detection/side_view/side_edge.jpg'
### path for the side iamge inception model############################
side_inception='/home/shorav/all_codes/inception/side_view/frame1.jpg'
side_inception2='/home/shorav/all_codes/inception/side_view/'
### path for the side image poet model################################
#side_poet='/home/shorav/all_codes/poet/side_view/side_poet.jpg'
#### path for the side image background subtraction####################
#side_background='/home/shorav/all_codes/background_subtraction/side_view/side_background.jpg'

#
#### label lines for the edges#################################
#label_top_edge=[line.rstrip() for line in tf.gfile.GFile("")]
#label_side_edge=[line.rstrip() for line in tf.gfile.GFile("")]
#
##### label lines for the top view of the inception model#####################
#label_top_inception=[line.rstrip() for line in tf.gfile.GFile("")]
#label_side_inception=[line.rstrip() for line in tf.gfile.GFile("")]
#
##### label lines for the top view of poet model##############################
#label_top_poet=[line.rstrip() for line in tf.gfile.GFile("")]
#label_side_poet=[line.rstrip() for line in tf.gfile.GFile("")]
#
###### label lines for the top view of background model#######################
#label_top_background=[label.rstrip() for line in tf.gfile.GFile("")]
#label_side_background=[label.rstrip() for line in tf.gfile.GFile("")]





label_lines = [line.rstrip() for line in tf.gfile.GFile("final_labels.txt")]

#label_lines = [line.rstrip() for line
#                   in tf.gfile.GFile("final_labels.txt")]
#######################################################################################
### Graph file read for edge
### top
#with tf.gfile.FastGFile("./top_edge.pb") as f:
#    graph_def=tf.GraphDef()
#    graph_def.ParseFromString(f.read())
#    _ = tf.import_graph_def(graph_def, name='')    
### Side
#with tf.gfile.FastGFile("./side_edge.pb") as f2:
#    graph_def2=tf.GraphDef()
#    graph_def2.ParseFromString(f2.read())
#    _ = tf.import_graph_def(graph_def2, name='')
#    
#### Graph file for inception Model
### top
with tf.gfile.FastGFile("./final_graph.pb") as f3:
    graph_def3=tf.GraphDef()
    graph_def3.ParseFromString(f3.read())
    _ = tf.import_graph_def(graph_def3, name='')
#### Side    
#with tf.gfile.FastGFile("./final_graph.pb") as f4:
#    graph_def4=tf.GraphDef()
#    graph_def4.ParseFromString(f3.read())
#    _ = tf.import_graph_def(graph_def, name='')

#### Graph file for poet model
### top
#with tf.gfile.FastGFile("./top_poet.pb") as f5:
#    graph_def5=tf.GraphDef()
#    graph_def5.ParseFromString(f5.read())
#    _ = tf.import_graph_def(graph_def5, name='')
### Side
#with tf.gfile.FastGFile("./side_poet.pb") as f6:
#    graph_def6=tf.GraphDef()
#    graph_def6.ParseFromString(f6.read())
#    _ = tf.import_graph_def(graph_def, name='')
#    
####### Graph file for the BAckground Subtraction
### Top    
#with tf.gfile.FastGFile("./top_background.pb", 'rb') as f7:## reading the graph file from the folder
#    graph_def7 = tf.GraphDef()                
#    graph_def.ParseFromString(f7.read())
#    _ = tf.import_graph_def(graph_def7, name='')
### Side
#with tf.gfile.FastGFile("./side_background.pb") as f8:
#    graph_def8=tf.GraphDef()
#    graph_def8.ParseFromString(f8.read())
#    _ = tf.import_graph_def(graph_def8, name='')

##########################################################################################
frame_number=0;
i=1;
temp=0
cnt=0

frame2=np.array([])
frame_left=np.array([])

frame2=np.ones((480,640),dtype='uint8')
frame_left=np.ones((480,640),dtype='uint8')

roi_value=[]
roi_value2=[]

frame_number2=[]
output=[]
while True:
    frame_number+=1;
    #print(frame_number)
    ret, frame = cam.read()## upper Camera
    ret, frame_side=cam2.read()## lower Camera
    #frame[1:100,:]=255
    #frame[:,1:50]=255
    frame2[0:(height),0:(width)]=255
    frame2[200:350,580:600]=0
    frame_left[200:350,200:220]=0
    if i<3:
        frame1=frame
        i+=1
    
    
    if ret== True:
        #im = cv2.imread(frame)
        imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        imgray1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,127,255,0)
        ret2,th2 = cv2.threshold(imgray,0,255,cv2.THRESH_OTSU)## for finding the otsu of the image
        #thresh[200:350,580:600]=0
        ## to count the total number of black pixel in the particular area
        roi=thresh[200:350,580:600]## this is same as the black strip on the right most part
        roi2=thresh[200:350,300:320]## this is the black strip for the left strip
        num_of_nonzero=np.count_nonzero(roi) # to count the number of pixel in the particular region
        num_of_nonzero2=np.count_nonzero(roi2)
        
        roi_value.append(num_of_nonzero)
        roi_value2.append(num_of_nonzero2)
        
        numpy_concat=np.concatenate((frame,frame_side),axis=1)

        
        
        #### to find the instance of the lego into the region
        if (frame_number>10) :
                if (roi_value2[frame_number-(5)]<2950)and(roi_value2[frame_number-(3)]<2950)and(roi_value2[frame_number-(1)]<2950)and(roi_value2[frame_number-(9)]<2950)and(roi_value2[frame_number-(8)]<2950):
                    if temp==0:   
                        frame_number2.append(frame_number)
                        cnt+=1
                        temp+=1
                        image_data = tf.gfile.FastGFile(top_inception, 'rb').read()
                        with tf.Session() as sess:
                                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
                                predictions = sess.run(softmax_tensor, \
                                         {'DecodeJpeg/contents:0': image_data})
                                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                                score_results=[]
                                final_label=[]
                                for node_id in top_k:
                                    human_string = label_lines[node_id]
                                    score = predictions[0][node_id]
                                    print(Fore.YELLOW +Back.BLUE+Style.BRIGHT+'%s (score = %.5f)' % (human_string, score)+Style.RESET_ALL)
                                    score_results.append(score)
                                    final_label.append(human_string)
                                output.append(final_label[0])
                                                                
                                
                        name = raw_input("Is Prediction Right or Wrong?\n For Right Press 'r'\n For Wrong Press 'w'")                                        
                        if name=='r':
                            
                            cv2.imwrite(os.path.join(top_inception2 ,'right_'+final_label[0]+str(frame_number)+'.jpg'),frame)
                            cv2.imwrite(os.path.join(side_inception2 ,'right_'+final_label[0]+str(frame_number)+'.jpg'),frame_side)

                        elif name=='w':
                            cv2.imwrite(os.path.join(top_inception2 ,'wrong_'+final_label[0]+str(frame_number)+'.jpg'),frame)
                            cv2.imwrite(os.path.join(side_inception2 ,'wrong_'+final_label[0]+str(frame_number)+'.jpg'),frame_side)


                        
                        with open('main.txt','w+') as g:
                            sd= '\n'.join(output)
                            g.write(sd)
                        
                else:
                    temp=0    
                
            
        print('frame number is :-%d \n value is:- %d'%(frame_number,num_of_nonzero))
        
        cv2.imshow('numpy_concat',numpy_concat)
        cv2.imshow('thresh',thresh)
        #cv2.imshow('Otsu', th2)
        #cv2.imshow('actual',frame)
        
###########Contour detection and drawing#########################################################
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)        
        img = cv2.drawContours(frame2, contours, -1, (0,255,0), 1)
        cv2.imshow('contour',frame2)
        
	if cv2.waitKey(30) & 0xFF==ord('q'):
	   break
    else:
	break
cam.release()
cv2.destroyAllWindows()

	           
