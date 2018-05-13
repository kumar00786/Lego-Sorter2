# -*- coding: utf-8 -*-
"""
This code covers following things:-
1. contour detection in video
2. Background subtraction in video
3. edge detection in video
4. Taking the snapshot at the particular instance
5. doing the Prediction with the help of tensor flow
6. Writing the prediction into the text file 
"""
import numpy as np
import cv2,time,os
from colorama import *
#cam= cv2.VideoCapture('/home/shorav/lego_proj/video/output1.avi')
cam= cv2.VideoCapture(0)
height = cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
width = cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
height=int(height)
width=int(width)
######### Tensor Flow##########################################################
'''
to implement the tensor flow following things are needed:-
1. Graph in the .pb format in the same folder
2. labels of all the category in the text file 
'''
import tensorflow as tf
image_path='/home/shorav/all_codes/frame1.jpg'
right_path='/home/shorav/all_codes/right_image/' # this is for saving the image which has the right prediction
wrong_path='/home/shorav/all_codes/wrong_image/' # this is for saving the image which has the wrong prediction
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("final_labels.txt")]
                   

with tf.gfile.FastGFile("./final_graph.pb", 'rb') as f:## reading the graph file from the folder
    graph_def = tf.GraphDef()                
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')



frame_number=0;
i=1;
temp=0
cnt=0
frame2=np.array([])
frame2=np.ones((480,640),dtype='uint8')
roi_value=[]
frame_number2=[]
output=[]
while True:
    frame_number+=1;
    #print(frame_number)
    ret, frame = cam.read()
    #frame[1:100,:]=255
    #frame[:,1:50]=255
    frame2[0:(height),0:(width)]=255
    frame2[200:350,580:600]=0
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
        roi=thresh[200:350,580:600]
        num_of_nonzero=np.count_nonzero(roi) # to count the number of pixel in the particular region
        roi_value.append(num_of_nonzero)        
        #### to find the instance of the lego into the region
        if (frame_number>10) :
                if (roi_value[frame_number-(5)]<2950)and(roi_value[frame_number-(3)]<2950)and(roi_value[frame_number-(1)]<2950)and(roi_value[frame_number-(9)]<2950)and(roi_value[frame_number-(8)]<2950):
                    if temp==0:   
                        frame_number2.append(frame_number)
                        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
                        #cv2.imwrite('frame'+str(frame_number)+'.jpg',frame)
                        cnt+=1
                        temp+=1
                        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
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
                            cv2.imwrite(os.path.join(right_path ,'right_'+final_label[0]+str(frame_number)+'.jpg'),frame)
                        elif name=='w':
                            cv2.imwrite(os.path.join(wrong_path ,'wrong_'+final_label[0]+str(frame_number)+'.jpg'),frame)
                        with open('main.txt','w+') as g:
                            sd= '\n'.join(output)
                            g.write(sd)
                        
                else:
                    temp=0    
                
            
        print('frame number is :-%d \n value is:- %d'%(frame_number,num_of_nonzero))
        
        cv2.imshow('thresh',thresh)
        #cv2.imshow('Otsu', th2)
        cv2.imshow('actual',frame)
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

	           
