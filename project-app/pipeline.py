from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from graphlab import SFrame
from graphlab import SArray
import graphlab
from app_pipeline import *
from numpy import unravel_index
import numpy as np
import psycopg2
import math
import cv2
import urllib2
import os
from PIL import Image

"""Extends the app_pipline to also featurize with GraphLab"""



def graph_lab(url, format = 'auto', flip_img = False, zoom = False):
	"""Extracts the graphlab features"""
	if format == 'auto':	
		extension = url.split('.')[-1]

	img = preprocess(url)
	if flip_img:
		img = flip(img)
	if zoom:
		img = middle(img)

	h,w,_ = img.shape
	img_bytes = bytearray(img)
	image_data_size = len(img_bytes)
	img = graphlab.Image(_image_data=img_bytes, _width=w, _height=h, _channels=3, _format_enum=2, _image_data_size=image_data_size)

	return SFrame({'image': [img]})


def extract_features(url, model, transformer, format = 'auto', flip_img = False, zoom = False):
	processed_images = transformer(url,format, flip_img, zoom)
	features = model.extract_features(processed_images)



if '__name__'=='__main__':
	"""Featurizes all the images in the dataset, and loads them into a database"""

	# The password for the database
	with open('~/Documents/Zipfian/project-app/.db_password.txt', 'rb') as f:
    	PASSWORD = f.readline()
	conn = psycopg2.connect("dbname='postgres' user='Tyler' host='zipfianproject.co9ijljkg9gd.us-east-1.rds.amazonaws.com' port='5432' password=%s")%PASSWORD
	cur = conn.cursor()

	#loaded the GraphLab Model
	pretrained_model = graphlab.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')


	pre_name = '../clothes_recommender/%s'
	conn.rollback()
	cur.execute('''SELECT images.id, dress_id, file_name, color
	                FROM images
	                JOIN clothes ON dress_id = clothes.id;''')
	gl_features = []
	gl_or = []

	top_colors = []
	color_features = []
	texture_features = []

	dress_ids = []
	colors = []
	img_ids = []

	bad_data = {}
	for img_id, dress, fil,color in cur.fetchall():
	    if not fil:
	        continue
	        
	    bools = (True, False)
	    for bool1 in bools:
	        for bool2 in bools:
	            fil_name = pre_name%fil

                gl = pipeline.extract_features(fil_name, pretrained_model, graph_lab, 
                                        format = 'JPG', flip_img = bool2, zoom = bool2)[0].tolist()
                
                gl_features.append(gl)

                gl_original = extract_features(fil_name, pretrained_model, original_graph_lab, 
                                            format = 'JPG', flip_img = bool2, zoom = bool2)[0].tolist()
                gl_or.append(gl_original)

                top_color = top_color(fil_name)
                top_colors.append(top_color)

                cols = color_process(fil_name, flip_img = bool2, zoom = bool2)
                color_features.append(cols)

                texture = texture_process(fil_name, flip_img = bool2, zoom = bool2)
                texture_features.append(texture)

                dress_ids.append(dress)
                img_ids.append(img_id+10000*bool1+20000*bool2)
                colors.append(color)

    #Load the features into a data frame and pickle it for later use
    df = pd.DataFrame({'id': img_ids, 'dress_ids': dress_ids, 'gl_features': gl_features, 
                   'color_features': color_features, 'texture_features': texture_features,
                  'top_colors': top_colors, 'gl_original': gl_or})
	pickle.dump(df,open('image_features','wb'), pickle.HIGHEST_PROTOCOL)

