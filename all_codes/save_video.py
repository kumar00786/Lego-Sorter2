import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap2=cv2.VideoCapture(1)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
out2 = cv2.VideoWriter('output2.avi', fourcc2, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    ret2, frame2=cap2.read()
    if ret==True:
        #frame = cv2.flip(frame,0)


        out1.write(frame)
        out2.write()

        cv2.imshow('frame',frame)
        cv2.imshow('frame2',frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

out.release()

cv2.destroyAllWindows()
