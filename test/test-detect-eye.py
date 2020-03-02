import cv2 as cv
import numpy as np

def main():
	# load-haar-classifiers
	face_haar_cascade = cv.CascadeClassifier('../assets/haarcascade-frontalface-default.xml')
	eye_haar_cascade = cv.CascadeClassifier('../assets/haarcascade-eye.xml')

	# load-image
	face_img = cv.imread('../assets/face.jpg')

	# convert-face-to-grayscale-for-face-detection
	gray_face_img = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)

	# detect-face
	face_detected = face_haar_cascade.detectMultiScale(gray_face_img, 1.3, 5)

	# draw-border-on-detected-face-and-for-eyes
	for (x, y, w, h) in face_detected:
		cv.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 0), 2)
		detected_gray_face_img = gray_face_img[y : y + h, x : x + h]
		detected_face_img = face_img[y : y + h, x : x + h]
		eyes_detected = eye_haar_cascade.detectMultiScale(detected_gray_face_img)
		for (ex, ey, ew, eh) in eyes_detected:
			cv.rectangle(detected_face_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

	cv.imshow('Face Detection', face_img)
	cv.waitKey(0)
	cv.destroyAllWindows()

if __name__ == '__main__':
	main()