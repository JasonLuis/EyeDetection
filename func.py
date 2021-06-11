import cv2 
import numpy as np
from math import hypot

def mid_point(p1, p2):
  return int((p1.x + p2.x)/2) , int((p1.y + p2.y)/2)

def get_ratio(eye_points, facial_landmarks):
  left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
  right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
  center_top = mid_point(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
  center_bottom = mid_point(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
  
  hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
  ver_line_lenght = hypot((center_top[0]- center_bottom[0]), (center_top[1] - center_bottom[1]))

  ratio = hor_line_lenght / ver_line_lenght

  return ratio

def get_gaze_ratio(frame,gray,eye_points, facial_landmarks):
    
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],np.int32)
    
    
    height, width, _ = frame.shape    
    mask = np.zeros((height,width), np.uint8)
   
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray,gray, mask=mask)
    
    min_x = np.min(left_eye_region[:,0])
    max_x = np.max(left_eye_region[:,0])
    min_y = np.min(left_eye_region[:,1])
    max_y = np.max(left_eye_region[:,1])
    
    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    
    right_side_threshold = threshold_eye[0: height, int(width/2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    
    if left_side_white == 0:
      gaze_ratio = 1
    elif right_side_white == 0:
      gaze_ratio = 5 
    else:
      gaze_ratio = left_side_white/right_side_white  
    return gaze_ratio

def drawer_eye(points, landmarks, frame, gray):
    eye_region = get_eye_region(points, landmarks)

    # mask
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)

    # eye mask
    cv2.polylines(mask, [eye_region], True, (255,255,255), 2)
    cv2.fillPoly(mask, [eye_region], (255,255,255))
    eye_mask = cv2. bitwise_and(gray, gray, mask=mask)

    # threshod binary of eye
    gray_eye,coordenadas = min_max_frame(eye_region, eye_mask)
    _, threshold_eye = cv2.threshold(gray_eye, 63, 255, cv2.THRESH_BINARY)

    #cv2.imshow(name, threshold_eye)
    return threshold_eye, gray_eye, coordenadas



def min_max_frame(region, frame):
    min_x = np.min(region[:, 0])
    max_x = np.max(region[:, 0])
    min_y = np.min(region[:, 1])
    max_y = np.max(region[:, 1])
    # print(min_x, max_x, min_y, max_y)
    l1 = [min_x,max_x,min_y,max_y]
    return frame[min_y: max_y, min_x: max_x],l1


def get_eye_region(points, landmarks):
    return np.array([(landmarks.part(points[0]).x, landmarks.part(points[0]).y),
                     (landmarks.part(points[1]).x, landmarks.part(points[1]).y),
                     (landmarks.part(points[2]).x, landmarks.part(points[2]).y),
                     (landmarks.part(points[3]).x, landmarks.part(points[3]).y),
                     (landmarks.part(points[4]).x, landmarks.part(points[4]).y),
                     (landmarks.part(points[5]).x, landmarks.part(points[5]).y)], np.int32)


def detect_iris(img,frame,eyes):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    gray_filtered = cv2.inRange(img_gray, 255, 255)
    contornos, _ = cv2.findContours(gray_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    cv2.drawContours(frame, contornos, -1, [0, 255, 0], 1);
    cv2.imshow(eyes, frame)
    return frame, contornos


def filter_color(img):
   """ retorna a imagem filtrada"""
   img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   #cv2.imshow('teste', img_gray)
   return img_gray 

def draw_cross(img, contorno):
  cnt=contorno
  M = cv2.moments(cnt)

  #desenhar a cruz
  size = 10
  color = (0,0,255)
  cX = 0
  cY = 0
  
  if M["m00"] != 0:
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
       
    cv2.line(img,(cX - size,cY),(cX + size,cY),color,1)
    cv2.line(img,(cX,cY - size),(cX, cY + size),color,1)
      
  return img
