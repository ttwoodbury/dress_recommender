import numpy as np
import psycopg2
from app_pipeline import euclidean
from pandas.io.sql import read_frame
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.decomposition import TruncatedSVD



class Recommender:
	def __init__(self, data):
		with open('~/Documents/Zipfian/project-app/.db_password.txt', 'rb') as f:
    		PASSWORD = f.readline()


		self.conn = psycopg2.connect("dbname='postgres' user='Tyler' host='zipfianproject.co9ijljkg9gd.us-east-1.rds.amazonaws.com' port='5432' password=%s")%PASSWORD
		self.cur = self.conn.cursor()

		self.similar_mask = None

		self.similar_titles = []
		self.similar_img_files = []
		self.similar_dress_urls = []
		self.similar_price = []


		self.image_data = data


	def predict(self, img_features, features, weights):
		"""Calculates the distances between the test image, and all the images in the database,
		weighting each feature vector seperately.  Stores the distances for later use"""

		if self.image_data.shape[0] == 0:
			return False

		df = self.image_data.copy()
	
		img_ids = df['id'].values

		distances = 0 
		for feature, weight in zip(features,weights):
			stored_features = np.array(df[feature].values.tolist())

			img_feature = np.array(img_features[feature])

			distances+=euclidean(stored_features,img_feature)*weight

		self.similar_mask = np.argsort(distances)

		return True


	def top_unique(self, k=5):
		"""Finds the top 'k' dresses associated with the closest images to the test image
		and queries the database for information about that dress"""

		dress_ids = self.image_data['dress_ids']
		similar = np.array(dress_ids)[self.similar_mask]
		similar_dresses = []
		index = 0

		num_dresses = len(dress_ids)
		k = min(k,num_dresses)

		while len(similar_dresses) < k:
			if similar[index] not in similar_dresses:
				similar_dresses.append(similar[index])
			index+=1

		self.cur.execute('''SELECT id, url FROM clothes WHERE id = ANY(%s);''',(similar_dresses,))
		ids, links = zip(*self.cur.fetchall())

		img_pre = 'https://s3.amazonaws.com/tylerzipfianproject/%s'
		for dress_id in ids:
			self.cur.execute('''SELECT file_name FROM images WHERE dress_id = %s AND file_name IS NOT NULL;'''%dress_id)
			self.similar_img_files.append(img_pre%self.cur.fetchone()[0])

			self.cur.execute('''SELECT url, title, price FROM clothes WHERE id = %s;'''%dress_id)
			result = self.cur.fetchone()
			self.similar_dress_urls.append(result[0])
			self.similar_titles.append(result[1])
			self.similar_price.append(result[2])
			


