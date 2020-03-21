import cv2 as cv
import numpy as np

def nothing(x):
	pass

def init_blob_detector():
	detector_params = cv.SimpleBlobDetector_Params()
	detector_params.filterByArea = True
	detector_params.maxArea = 1500 
	detector = cv.SimpleBlobDetector_create(detector_params)
	return detector

def init_haar_classifiers():
	face_haar_classifier = cv.CascadeClassifier('../assets/haarcascade-frontalface-default.xml')
	eye_haar_classifier = cv.CascadeClassifier('../assets/haarcascade-eye.xml')
	return face_haar_classifier, eye_haar_classifier

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
		detected_face_coordinates = (x, y, w, h)

	return detected_face_region, detected_face_coordinates

def detect_eyes(img, eye_haar_classifier):
	# convert image to grayscale
	gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	# extract all the eye coordinates
	eye_coordinates = eye_haar_classifier.detectMultiScale(gray_img, 1.3, 5)

	# initalize variables
	left_eye_coordinates = None
	rigth_eye_coordinates = None
	width = np.size(img, 1)
	height = np.size(img, 0)

	# remove false postive and detect left and right eye coordinates
	for (x, y, w, h) in eye_coordinates:    
		if y > height / 2:
			pass
		eye_center = (x + w) / 2
		if eye_center < width / 2:
			left_eye_coordinates = (x, y, w, h)
		else:
			rigth_eye_coordinates = (x, y, w, h)

	return left_eye_coordinates, rigth_eye_coordinates

def main():
	cap = cv.VideoCapture(0)
	cv.namedWindow('image')
	# get-blob-detector
	detector = init_blob_detector()
	face_haar_classifier, eye_haar_classifier = init_haar_classifiers()
	while True:
		_, img = cap.read()
		if img is not None:
			face, face_coordinates = detect_face(img, face_haar_classifier)
		else:
			break
		print (face)
		print (face_coordinates)
		if face is not None and face_coordinates is not None:
			eyes_coordinates = detect_eyes(face, eye_haar_classifier)
			print (eyes_coordinates)
			for e in eyes_coordinates:
				if e is not None:
					x = face_coordinates[0] + e[0]
					y = face_coordinates[1] + e[1]
					w = face_coordinates[2] + e[2]
					h = face_coordinates[3] + e[3]
					cv.rectangle(face, (x, y), (x + w, y + h), (0, 0, 255), 2)				
			cv.imshow('detected-eyeball', face)
			if cv.waitKey(1) & 0xFF == ord('q'):
				break
	
	cap.release()
	cv.destroyAllWindows()

if __name__ == '__main__':
	main()
