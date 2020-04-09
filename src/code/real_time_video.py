import cv2
import numpy as np 
from keras.models import model_from_json
from keras.preprocessing import image

#load saved model
model = model_from_json(open("../models/fer.json","r").read())
model.load_weights("fer.h5")

#emotions
emotions = ('angry','disgust','fear','happy','sad','surprise','neutral')

#openCV face detector
face_haar_cascade = cv2.CascadeClassifier('../haarcascade_files/haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

while True:
	ret,img = cam.read()
	if not ret:
		continue
	gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces_detected = face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=5)

	if len(faces_detected) :
		for (x,y,w,h) in faces_detected:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			crop_gray = gray_img[y:y+h,x:x+w] #cropping out Region Of Interest
			#cv2.imshow("cropped_image",crop_gray)
			crop_gray = cv2.resize(crop_gray,(48,48))
			img_pixels = image.img_to_array(crop_gray)
			#print("shape:",img_pixels.shape)
			img_pixels = np.expand_dims(img_pixels,axis=0)
			#print("shape:",img_pixels.shape)
			img_pixels /= 255

			predictions = model.predict(img_pixels)

			max_predicted_val_index = np.argmax(predictions[0])

			predicted_emotion =emotions[max_predicted_val_index]

			cv2.putText(img,predicted_emotion,(int(x),int(y+h+10)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) 

		resized_img = cv2.resize(img,(750,750))
		cv2.imshow('Face Emotion Analysis',resized_img)
	else:
		print("No Faces detected\n");

	if cv2.waitKey(10) == ord('q'):
		break

cam.release()
cam.destroyAllWindows()