import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import winsound 
import time
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 12000))
Flag = True
sock.listen(1)
conn, addr = sock.accept()

video_capture = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

right_eye_points = list(range(36, 42))
left_eye_points = list(range(42, 48))

start_time = time.process_time()
count_time4 = 0
count_time5 = 0
count = 0
undetection_time = 0
running_time = 0
carelessness_state = False 
drowsiness_state = False
drowsiness_state2 = False
awakeness_state = False
undetection_state = False
Limit_EAR_left = 0
Limit_EAR_right = 0

left = []
right = []
current_state = 'waiting'

def eye_ratio(eyepoint):
    A = dist.euclidean(eyepoint[1],eyepoint[5])
    B = dist.euclidean(eyepoint[2],eyepoint[4])
    C = dist.euclidean(eyepoint[0],eyepoint[3])
    EAR = (A+B) / (2.0*C)
    return EAR
    
while True:    
   ret, frame = video_capture.read()
   frame2 = cv2.flip(frame, 1)
   gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   clahe_image = clahe.apply(gray)
   detection = face_detector(clahe_image)
   key = cv2.waitKey(20) & 0xff
   
   for d in detection:
       
       shape = shape_predictor(clahe_image, d) 
       landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
            
       left_eye = landmarks[left_eye_points]
       right_eye = landmarks[right_eye_points]
        
       EAR_left = float("{0:.2f}".format(eye_ratio(left_eye)))
       EAR_right = float("{0:.2f}".format(eye_ratio(right_eye)))
       
       left_eye_average_x = int((shape.part(36).x + shape.part(39).x)/2) 
       left_eye_average_y = int((shape.part(36).y + shape.part(39).y)/2) 
       right_eye_average_x = int((shape.part(42).x + shape.part(45).x)/2) 
       right_eye_average_y = int((shape.part(42).y + shape.part(45).y)/2) 
       
       line = float("{0:.2f}".format(100*((shape.part(30).x - shape.part(27).x)/(shape.part(30).y - shape.part(27).y))))
             
       rob = gray[shape.part(45).y, shape.part(45).x]
       rpb = gray[right_eye_average_y, right_eye_average_x] 
       rib = gray[shape.part(42).y, shape.part(42).x]
       
       lob = gray[shape.part(39).y, shape.part(39).x]
       lpb = gray[left_eye_average_y, left_eye_average_x]  
       lib = gray[shape.part(36).y, shape.part(36).x]
       
       cv2.circle(frame2, (shape.part(30).x, shape.part(30).y), 1, (0,255,255), thickness=1)
       cv2.circle(frame2, (shape.part(27).x, shape.part(27).y), 1, (0,255,255), thickness=1)
       cv2.line(frame2, (shape.part(30).x, shape.part(30).y), (shape.part(27).x, shape.part(27).y), (0,255,0), 1)
      
       cv2.circle(frame2, (right_eye_average_x, right_eye_average_y), 1, (0,0,255), thickness=1)
       cv2.circle(frame2, (left_eye_average_x, left_eye_average_y), 1, (0,0,255), thickness=1)
       
       for i in range(36, 48):
           cv2.circle(frame2, (shape.part(i).x, shape.part(i).y), 1, (255,255,0), thickness=1)
              
       if current_state == 'waiting':
           cv2.putText(frame2, "Press p to calculate your EAR", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
           cv2.putText(frame2, "Press q to end the program", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
           count_time2 = time.process_time()-start_time
           
       if current_state == 'running':
         
           count_time3 = time.process_time()-start_time
                      
           if undetection_state == True:
               winsound.PlaySound(None, winsound.SND_ASYNC)
               undetection_state = False
               
           running_time = int(time.process_time()-start_time-count_time2)
           
           if running_time < 4:
               cv2.putText(frame2, "calculating...", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
               cv2.putText(frame2, "Wait for seconds..", (150, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
               cv2.putText(frame2, "running_time : {0:d}".format(running_time), (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
               left.append(EAR_left)
               right.append(EAR_right)
               count = count + 1
               
           if running_time < 6:
               count_time5 = time.process_time()-start_time
               count_time = time.process_time()-start_time
           
           Sum_EAR_left = float("{0:.2f}".format(sum(left)/count))
           Sum_EAR_right = float("{0:.2f}".format(sum(right)/count)) 
           
           Limit_EAR_left = float("{0:.2f}".format(0.8*Sum_EAR_left))
           Limit_EAR_right = float("{0:.2f}".format(0.8*Sum_EAR_right))
                
           if (line > -15 and line < 5) and (rpb < rib and rpb < rob and lpb < lib and lpb < lob):
               count_time5 = time.process_time()-start_time
               
           carelessness_time = float("{0:.1f}".format(time.process_time()-start_time-count_time5)) 
           
           if carelessness_time == 3.0:
               winsound.PlaySound("alarm.wav", winsound.SND_LOOP + winsound.SND_ASYNC)
               carelessness_state = True
               
           if carelessness_state == True:    
               cv2.putText(frame2, "!!!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
               if (line > -15 and line < 5) and (rpb < rib and rpb < rob and lpb < lib and lpb < lob):
                   carelessness_state = False
                   winsound.PlaySound(None, winsound.SND_ASYNC)
           
           if drowsiness_state == False and drowsiness_state2 == False:               
               if EAR_left > Limit_EAR_left and EAR_right > Limit_EAR_right: 
                   count_time = time.process_time()-start_time
                                               
           drowsiness_time = float("{0:.1f}".format(time.process_time()-start_time-count_time))
           
           if drowsiness_time == 1.5:
               winsound.PlaySound("alarm.wav", winsound.SND_LOOP + winsound.SND_ASYNC)
               drowsiness_state2 = True
               carelessness_state = False
               
           if awakeness_state == False:
               count_time4 = time.process_time()-start_time
                
           awakeness_time = float("{0:.1f}".format(time.process_time()-start_time-count_time4))  
           print(awakeness_time)
                              
           if drowsiness_state2 == True:
               count_time5 = time.process_time()-start_time
               cv2.putText(frame2, "!!!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
               if EAR_left > Limit_EAR_left and EAR_right > Limit_EAR_right:
                   awakeness_state = True
               else:
                   awakeness_state = False
                              
           if awakeness_time == 1.0 and drowsiness_time < 4.0:              
               drowsiness_state2 = False
               awakeness_state = False
               winsound.PlaySound(None, winsound.SND_ASYNC)
                          
           if drowsiness_time == 4.0:
               drowsiness_state = True
               drowsiness_state2 = False
               awakeness_state = False
                              
           if drowsiness_state == True:
               count_time5 = time.process_time()-start_time
               cv2.putText(frame2, "press p to turn off ALARM", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2) 
               cv2.putText(frame2, "!!!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
               
           if running_time > 4:  
               cv2.putText(frame2, "running...", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)  
               cv2.putText(frame2, "drowsiness_time : {0:.1f}".format(drowsiness_time), (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
               cv2.putText(frame2, "carelessness_time : {0:.1f}".format(carelessness_time), (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
               cv2.putText(frame2, "EAR_R : {0:.2f}".format(Limit_EAR_right), (10, 400),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),2)
               cv2.putText(frame2, "EAR_L : {0:.2f}".format(Limit_EAR_left), (10, 420),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),2)
               cv2.putText(frame2, "EAR_R2 : {0:.2f}".format(EAR_right), (450, 400),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),2)
               cv2.putText(frame2, "EAR_L2 : {0:.2f}".format(EAR_left), (450, 420),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),2)
               
           cv2.putText(frame2, "press q to end the program", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
               
   if not detection: 
       
       if current_state == 'running':
           cv2.putText(frame2, "detection nothing...", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
           cv2.putText(frame2, "undetection_time : {0:.1f}".format(undetection_time), (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
           cv2.putText(frame2, "press q to end the program", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
           
           if drowsiness_state == False and carelessness_state == False:
               count_time = time.process_time()-start_time
               count_time5 = time.process_time()-start_time
               undetection_time = float("{0:.1f}".format(time.process_time()-start_time-count_time3))
               
       if undetection_time >= 7.0:
           cv2.putText(frame2, "!!!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
       if undetection_time == 7.0:
           winsound.PlaySound("alarm.wav", winsound.SND_LOOP + winsound.SND_ASYNC)
           undetection_state = True

   if drowsiness_state == True:
       sendlen = conn.send(bytes("False","utf-8"))
       
   if current_state == 'running':
       if drowsiness_state == False:
            sendlen = conn.send(bytes("True","utf-8"))
             
   if key == ord('q'):
       winsound.PlaySound(None, winsound.SND_ASYNC)
       break
   
   if key == ord('p'):
       if detection:
           if current_state == 'waiting':
               current_state = 'running'
               
           if current_state == 'running':
               if drowsiness_state == True:
                   if EAR_left > Limit_EAR_left and EAR_right > Limit_EAR_right:
                       drowsiness_state = False
                       winsound.PlaySound(None, winsound.SND_ASYNC)
                   
   cv2.imshow("Frame", frame2)

conn.close()       
sock.close()
cv2.destroyAllWindows()
video_capture.release()
