import cv2
import numpy as np

from keras.models import load_model from models.cnn import mini_XCEPTION import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session config = tf.compat.v1.ConfigProto()
input_shape = (48,48,1)
num_classes = 7
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') # using the Haar Cascade as classifier to detect frontal face video_capture = cv2.VideoCapture(0)
# using function in cv2 to start capturing video
model = mini_XCEPTION(input_shape, num_classes) model.load_weights('final_model.h5') #model.load("model_5-49-0.62.hdf5") model.get_config()
target = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'] font = cv2.FONT_HERSHEY_SIMPLEX

# font in which test is to be displayed while True:
# Capture frame-by-frame
ret, frame = video_capture.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# converting the image into gray scale image as accuracy of morphological features is very high
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1)


# Draw a rectangle around the faces for (x, y, w, h) in faces:
cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 5) face_crop = frame[y:y + h, x:x + w]
face_crop = cv2.resize(face_crop, (48, 48))

face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) face_crop = face_crop.astype('float32') / 255
face_crop = np.asarray(face_crop)
face_crop = face_crop.reshape(1,face_crop.shape[0], face_crop.shape[1],1) print(model.predict(face_crop)) print(np.argmax(model.predict(face_crop)))
result = target[np.argmax(model.predict(face_crop))] #result to be displayed i.e. the emotion
cv2.putText(frame, result, (x, y), font, 1, (200, 0, 0), 3, cv2.LINE_AA) # frame of output with the text written over that
print(result)
if (result=="fear"): import smtplib, ssl port = 465 # For SSL
smtp_server = "smtp.gmail.com" sender_email = "" # Enter your address receiver_email = "" # Enter receiver address password="" #Enter your password
message = """


Subject: Fear Detected! Please send help


Hello! Passenger Safety System detected fear from live video stream from pramods ri de. Requesting help ASAP!!"""

context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server: server.login(sender_email, password)
server.sendmail(sender_email, receiver_email, message) # Display the resulting frame
else:
continue

cv2.imshow('Video', frame)
if cv2.waitKey(1) & 0xFF == ord('q'): break
# When everything is done, release the capture video_capture.release() cv2.destroyAllWindows()
