import cv2 as cv
import numpy as np

# global variable
detector = None

def init_blob_detector():
    detector_params = cv.SimpleBlobDetector_Params()
    detector_params.filterByArea = True
    detector_params.maxArea = 1500 
    detector = cv.SimpleBlobDetector_create(detector_params)

def detect_face(img, face_haar_classifier):
    # convert img to grayscale
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # extract the face coordinates using haar cascade
    face_coordinates = face_haar_classifier.detectMultiScale(gray_img, 1.3, 5)

    # find biggest detected region from the coordinates
    if len(face_coordinates) > 1:
        biggest_region_detected = (0, 0, 0, 0)
        # iterate for every detected region
        for i in face_coordinates:
            if i[3] > biggest_region_detected[3]:
                biggest_region_detected = i
            
        biggest_region_detected = np.array([i], np.int32)
    # if only one region detected the it is the face
    elif len(face_coordinates) == 1:
        biggest_region_detected = face_coordinates
    # if no region is detected return null
    else:
        return None
    
    # extract the face from the image using coordinates
    for (x, y, w, h) in biggest_region_detected:
        detected_face_region = img[y : y + h, x : x + w]

    return detected_face_region

def detect_eye(img, eye_haar_classifier):
    # convert image to grayscale
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # extract all the eye coordinates
    eye_coordinates = eye_haar_classifier.detectMultiScale(gray_img, 1.3, 5)

    # initalize variables
    left_eye = None
    right_eye = None
    width = np.size(img, 1)
    height = np.size(img, 0)

    # remove false postive and detect left and right eye
    for (x, y, w, h) in eye_coordinates:    
        if y > height / 2:
            pass
        eye_center = (x + w) / 2
        if eye_center < width / 2:
            left_eye = img[y : y + h, x : x + w]
        else:
            right_eye = img[y : y + h, x : x + w]

    return left_eye, right_eye

def remove_eyebrows():
    pass

def detect_eye_ball():
    pass
