import numpy as np
import cv2 as cv

def detect_eyes(face_img, eye_haar_cascade):

	left_eye = []
	right_eye = []

	gray_face_img = cv.cvtColor(face_img, cv.BGR2GRAY)
	eye_coordinates = eye_haar_cascade.detectMultiScale(gray_face_img, 1.3, 5)

	height = np.size(face_img, 0)
	width = np.size(face_img, 1)

	for  (x, y, w, h) in eye_coordinates:
		if y > height / 2:
			pass
		eye_center = (x + w) / 2
		if eye_center < width * 0.5:
			left_eye = face_img[y : y + h, x : x + w]
		else:
			right_eye = face_img[y : y + h, x : x + w]

	return left_eye, right_eye

def detect_face(face_img, face_haar_cascade):

	gray_face_img = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
	face_coordinates = face_haar_cascade.detectMultiScale(gray_face_img, 1.3, 5)
	
	# find-the-biggest-region
	if len(face_coordinates) > 1:
		biggest_region_detected = (0, 0, 0, 0)
		for i in face_coordinates:
			if i[3] > biggest_region_detected[3]:
				biggest_region_detected = i
		biggest_region_detected = np.array([i], np.int32)
	elif len(face_coordinates) == 1:
		biggest_region_detected = face_coordinates
	else:
		return None	
	
	for (x, y, w, h) in biggest_region_detected:
		detected_face_region = face_img[y : y + h, x : x + w]
	
	return detected_face_region