# Emotion-Recognition

# Introduction
This repository demonstrates prediction of human face emotion both in real time video and in an offline image using openCV and keras CNN in python. 
This project is trained to detect *angry*,*disgust*,*fear*,*happy*,*sad*,*surprise*,*neutral* emotions.

### 1. pre-processing 
I have used [this](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) dataset.</br>
Download this and place inside [this folder](https://github.com/vamc-stash/Emotion-Recognition/tree/master/src/fer2013/fer2013).</br>
segement the dataset into training and testing samples.

### 2. CNN model
Train the dataset using keras CNN and store the resultant model. Obtained model gives accuracy of 0.5818 after running for 60 epochs on 28709 images.</br>
> run python3 [cnn.py](https://github.com/vamc-stash/Emotion-Recognition/blob/master/src/code/cnn.py)

### 3. Real time video
This program will create a window to display the scene capturing by webcam and detect the faces using HAAR features and predicts facial emotion using pre-trained CNN model. </br>
> run python3 [real_time_video.py](https://github.com/vamc-stash/Emotion-Recognition/blob/master/src/code/real_time_video.py)

### 4. Offline image
This program takes an image as input and process the input to detect faces using HAAR features. Finally, predicts facial emotion using pre-trained CNN model.</br>
> run python3 [static_image.py](https://github.com/vamc-stash/Emotion-Recognition/blob/master/src/code/static_image.py)

# Installations
`numpy` `pandas` `cv2` `keras` 

# Acknowledgments
https://www.edureka.co/blog/convolutional-neural-network/ </br>
https://medium.com/themlblog/how-to-do-facial-emotion-recognition-using-a-cnn-b7bbae79cd8f




