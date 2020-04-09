import sys
import cv2
import numpy as np 
from keras.models import model_from_json
from keras.preprocessing import image

#load saved model
model = model_from_json(open("../models/fer.json","r").read())
model.load_weights("fer.h5")

#emotions
emotions = ('angry','disgust','fear','happy','sad','surprise','neutral')

def detect_face(img):
	gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#openCV face detector
	face_haar_cascade = cv2.CascadeClassifier('../haarcascade_files/haarcascade_frontalface_default.xml')
	faces_detected = face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=5)

	return faces_detected,gray_img


if __name__ == "__main__":
	image_name = sys.argv[1]
	image_path = "../test_images/"+image_name 
	img = cv2.imread(image_path)

	cv2.imshow("Testing Image : ",img)
	cv2.waitKey(100)

	faces,gray_img = detect_face(img)

	if len(faces):
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			crop_gray = gray_img[y:y+h,x:x+w]
			crop_gray = cv2.resize(crop_gray,(48,48))
			img_pixels = image.img_to_array(crop_gray)
			img_pixels = np.expand_dims(img_pixels,axis=0)
			img_pixels /= 255

			predictions = model.predict(img_pixels)
			max_predicted_val_index = np.argmax(predictions[0])
			predicted_emotion =emotions[max_predicted_val_index]

			cv2.putText(img,predicted_emotion,(int(x),int(y+h+10)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) 

		resized_img = cv2.resize(img,(750,750))
		cv2.imshow('Face Emotion Analysis',resized_img)
	else:
		print("No Faces detected\n");

	while(1):
		if cv2.waitKey(10) == ord('q'):
			break;
	
	cv2.destroyAllWindows()
	sys.exit()




	 

