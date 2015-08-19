import matplotlib.pyplot as plt
from numpy import unravel_index
import numpy as np
import math
import cv2
import urllib2
import os
from PIL import Image

"""Contains all the functions to process the images"""


def euclidean(x_array,y):
    return np.sum(np.power(x_array-y,2), axis =1)

def top_color(url):
	""" Finds the dominent color in the image"""
	img = preprocess(url)
	img = middle(img)

	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist( [hsv], [0,1], None, [180, 256], [0, 180, 1, 256] ).flatten()

	hist = hist.reshape(180,256)

	h,s = unravel_index(hist.argmax(), hist.shape)

	hsv[:,:,0] = h
	hsv[:,:,1] = s
	hsv[:,:,2] = 150
	#test = test.astype(int)

	img2 = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
	
	return tuple(img2[0,0,:])


def henry_viii(img):
	"""Uses the haar cascade classifier from OpenCV detect and remove faces"""
	face_cascade = cv2.CascadeClassifier('face_detector/haarcascade_frontalface_default.xml')
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 2)
	for face in faces:
		x,y,w,h = enlarge_rectangle(0,*face)
		cv2.rectangle(img,(0,0),(img.shape[1],y+h),(255,255,255),-1)

	return img


def background_substractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bool = gray/180 == 1
    for i in range(3):
        img[:,:,i][img_bool] = 255
    return img


def resize(img):
	h, w, _ = img.shape
	r = min(h,w)/2
	img = img[h//2-r:h/2.+r, w/2.-r:w/2.+r]
	img = Image.fromarray(img)

	rsize = img.resize((256,256))
	im = np.asarray(rsize)

	return im

def middle(img):
	"""Returns the middle of the picture"""
	h, w, _ = img.shape
	return img[h/2-50:h/2+50,w/2-50:w/2+50]


def enlarge_rectangle(incPct, x, y, w, h):
	"""Helper function to remove the faces from the images"""
	r = (w**2+h**2)**(1./2)/2
	theta = math.atan2(h,w)
	dx = int(r*incPct*math.cos(theta))
	dy = int(r*incPct*math.sin(theta))
    
	return x - dx, y - dy, w + 2*dx,h + 2*dy

def preprocess(url):
	"""Finds the image at the url, removes the heads,
	takes away the background, and resizes"""
	req = urllib2.urlopen(url)
	arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
	img = cv2.imdecode(arr,-1)
	#img = cv2.imread(url,-1)
	img = henry_viii(img)
	img = background_substractor(img)
	img = resize(img)

	return img

def color_process(url, flip_img = False, zoom = False):
	"""Creates a feature vector from the color histogram"""
	threshold = 240

	img = preprocess(url)
	if flip_img:
		img = flip(img)
	if zoom:
		img = middle(img)

	chans = cv2.split(img)
	features = {}
	img_features = []
	for i, chan in enumerate(chans):
		hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
		flat = hist.flatten()
		hist_chopped = flat[:threshold]/threshold
		features[i] = hist_chopped.tolist()

	result = features[0]
	result.extend(features[1])
	result.extend(features[2])
	return result

def extract_all_features(url, scaler):
	"""For use in the app, takes in a url, and
	does all the featurization.  Uses the scalers that 
	where used to normalized the images in the dataset"""

	features = {}

	color_features = color_process(url)
	features['color_pca'] = scaler.scale_color(color_features)


	max_color = top_color(url)
	features['top_scaled'] = scaler.scale_top(max_color)


	return features 

