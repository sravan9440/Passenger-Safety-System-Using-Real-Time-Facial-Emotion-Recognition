import pandas as pd import cv2
import numpy as np


#setting the path where our datset is located 
dataset_path = 'fer2013/fer2013/fer2013.csv' image_size=(48,48)

def load_fer2013():
data = pd.read_csv(dataset_path) #reading the csv file
pixels = data['pixels'].tolist() #converting pixel values into a list width, height = 48, 48
faces = []
#converting pixel values from each row into a 48x48 matrix of float values and appendin g it to the faces array
for pixel_sequence in pixels:
face = [int(pixel) for pixel in pixel_sequence.split(' ')] 
face = np.asarray(face).reshape(width, height)
face = cv2.resize(face.astype('uint8'),image_size) 
faces.append(face.astype('float32'))
faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)
emotions = pd.get_dummies(data['emotion']).to_numpy() #storing values from 0- 6 in emotion to an array
return faces, emotions


def preprocess_input(x, v2=True): 
x = x.astype('float32')
x = x / 255.0 # converting pixel values in the range 0-255
if v2:	# rescaling the image in the range of [-1,1] from [0,1] 
x = x - 0.5
x = x - 0.5
x = x * 2.0 
return x
