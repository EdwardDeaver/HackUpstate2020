
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
from tensorflow import keras
import smtplib
import datetime
import config
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

from dotenv import load_dotenv
import os
load_dotenv()
import cv2
from tensorflow.keras.preprocessing import image

## OPENCV LOAD from remote cam

#cap = cv2.VideoCapture(os.getenv("CamRTSPURL"), cv2.CAP_FFMPEG )
cap = cv2.VideoCapture(0)
OldTimeInSeconds2 =  datetime.datetime.now()

### VARs for EMAIL
sent_from = os.getenv("mailFromAddress")
to = [os.getenv("mailToAddress")]

## VARs for Keras Model
labels = ["Fedex", "UPS", "PEOPLE"]

## Tell TensorFlow I'm not using GPU acceleration. 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
# Load the model
model = tf.keras.models.load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)



##################################################
## Tensor Flow Analyzes the image passed to it 
##################################################
def tensorFlowAnalyze(image ):
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
  
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    #image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    indexOfPredictionMax = prediction[0].argmax()
    print(labels[indexOfPredictionMax])
    print(prediction[0].argmax())
    ## Only if it's a UPS or Fedex
    if(labels[indexOfPredictionMax] == "Fedex" or labels[indexOfPredictionMax] == "UPS"):
        sendEmail(labels[indexOfPredictionMax])
        time.sleep(10)

#####################################################################
## Modified from : https://github.com/code-and-dogs/pythonMail/blob/master/config.py
## Sends email via Gmail SMTP
## Truck here - string of label of truck
#####################################################################
def sendEmail(TruckHere):
    timeObject = datetime.datetime.now()
    timestampStr = timeObject.strftime("%d-%b-%Y (%H:%M:%S.%f)")

    body = TruckHere + "is here at " + timestampStr
    subject = "Your " + TruckHere + " is here"

    emailMessage = MIMEMultipart()
    emailMessage['From'] = sent_from
    emailMessage['To'] = os.getenv("mailToAddress")
    emailMessage['Subject'] = subject

    emailMessage.attach(MIMEText(body,'plain'))
    message = emailMessage.as_string()

    try:
        server = smtplib.SMTP(os.getenv("mailServer"))
        server.starttls()
        server.login(sent_from, os.getenv("mailFromPassword"))

        server.sendmail(sent_from, to, message)
        server.quit()
        print("SUCCESS - Email sent")

    except Exception as e:
        print("FAILURE - Email not sent")
        print(e)

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
'''
image = Image.open('test_photo0.jpg')
tensorFlowAnalyze(image)

image = Image.open('test_photo1.jpg')
tensorFlowAnalyze(image)

image = Image.open('test_photo2.jpg')
tensorFlowAnalyze(image)

'''

def getCVData():
    ret, frame = cap.read()
    cv2.imwrite('./pic.jpg', frame)
    image = Image.open('./pic.jpg')
    cv2.imshow("preview",frame)
    tensorFlowAnalyze(image)


previous = time.time()
delta = 0

while True:
    try:
        getCVData()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print("Error")
        print(e)

cap.release()
cv2.destroyAllWindows()

    #cv2.imwrite("frame.jpg", frame) 
    #image = Image.open('frame.jpg')

