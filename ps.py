import cv2 
import dlib
from imutils import face_utils
import numpy as np
import func

left_eye = [36,37,38,39,40,41]
right_eye = [42,43,44,45,46,47]
font = cv2.FONT_HERSHEY_PLAIN

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        face_landmarks = predictor(gray, face)
        #Drawer Eyes
        for n in range(36, 48):

            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y

            cv2.circle(frame, (x, y), 3, (0, 255, 255), 1)

            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]


            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            left_gaze_ratio, pointLeft, pl = func.drawer_eye(left_eye, face_landmarks, frame, gray)
            right_gaze_ratio, pointRight, pr = func.drawer_eye(right_eye, face_landmarks, frame, gray) 

            eyeLeft = func.filter_color(left_gaze_ratio)
            frame[pl[2]: pl[3], pl[0]: pl[1]], cont_left = func.detect_iris(eyeLeft,frame[pl[2]: pl[3], pl[0]: pl[1]],"Olho esquerdo")

            eyeRight = func.filter_color(right_gaze_ratio)
            frame[pr[2]: pr[3], pr[0]: pr[1]], cont_right = func.detect_iris(eyeRight,frame[pr[2]: pr[3], pr[0]: pr[1]],"Olho direito")

            if(len(cont_left)!= 0 or len(cont_right)!= 0):
                frame[pl[2]: pl[3], pl[0]: pl[1]] = func.draw_cross(frame[pl[2]: pl[3], pl[0]: pl[1]],cont_left[0])
                frame[pr[2]: pr[3], pr[0]: pr[1]] = func.draw_cross(frame[pr[2]: pr[3], pr[0]: pr[1]],cont_right[0])
            
        
        #Detect moviment
        gaze_ratio_left_eye = func.get_gaze_ratio(frame,gray,[36,37,38,39,40,41],face_landmarks)
        gaze_ratio_right_eye = func.get_gaze_ratio(frame,gray,[42,43,44,45,46,47],face_landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2    
        
        new_frame = np.zeros((600, 800, 3), np.uint8)
        new_frame[:] = (255,255,255)
        cv2.imshow("Face Landmarks", frame)
        
        if gaze_ratio <= 1:
            cv2.putText(new_frame, "ESQUERDA", (50,100), font, 2, (0,0,255),3)
       
        elif 1 < gaze_ratio < 3:
            cv2.putText(new_frame, "CENTRO", (50,100), font, 2, (0,0,255),3)  
        else:
            cv2.putText(new_frame , "DIREITA", (50,100), font, 2, (0,0,255),3)

        # Show direction
        cv2.imshow("Direção do Olho", new_frame)
        
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()


