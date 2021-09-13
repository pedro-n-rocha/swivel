import argparse

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import requests
import cv2
import numpy as np
import pickle

from keras.models import load_model
import imutils
import collections

parser = argparse.ArgumentParser()
parser.add_argument("--pin",required=True)
parser.add_argument("--pwd",required=True)
parser.add_argument("--usr",required=True)
parser.add_argument("--url",required=True)

args = parser.parse_args()

pin = [int(char) for char in args.pin]
pwd = args.pwd
usr = args.usr
url = args.url+"?username="+usr

fpwd = ""
code = [] 
solver = [] 

response = requests.get(url, stream=True)
arr = np.asarray(bytearray(response.content), dtype=np.uint8)
img = cv2.imdecode(arr, -1) 
#cv2.imshow('code', img)

c = img[38:71 , 9:289]
#cv2.imshow("c", c)
g = cv2.cvtColor(np.array(c), cv2.COLOR_BGR2GRAY)
(thresh, bw) = cv2.threshold(g, 60, 255, cv2.THRESH_BINARY)

#cv2.imshow("bw", bw)
with open('./model/model_labels.dat', "rb") as f:
    lb = pickle.load(f)
model = load_model('./model/model_cnn.hdf5')

def resize_to_fit(image, width, height):
    (h, w) = image.shape[:2]
    if w > h:
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image, height=height)
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))
    return image

for i in range(10):
  n = bw[0:, i*28:(i+1)*28]
  img = resize_to_fit(n, 20, 20)
  img = np.expand_dims(img, axis=2)
  img = np.expand_dims(img, axis=0)
  prediction = model.predict(img)
  number = lb.inverse_transform(prediction)[0]
  code.append(number) 

co = collections.deque(code)
co.rotate(1)
for p in pin:
  solver.append(co[p])
  fpwd += str(co[p])

print("\n\nfinal password: "+pwd+fpwd+"\n\n")

#cv2.waitKey(0)
