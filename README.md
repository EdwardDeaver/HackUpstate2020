# Is the mail here yet

Uses TensorFlow + OpenCV to check if the Fedex or UPS truck is here, and sends me an email if it is. 

ENV File parameters

mailToAddress = Address you want to send to
mailServer = "smtp.gmail.com:587" Gmails SMTP server
mailFromAddress =  Address you are sending emails from
mailFromPassword = Address you are sending emails password 
CamRTSPURL = RTSP server url if you want to use it


Requirments: 


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
import cv2
from tensorflow.keras.preprocessing import image