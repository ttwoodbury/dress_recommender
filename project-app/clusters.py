
from nltk.stem.porter import PorterStemmer
from string import punctuation
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TextClusters:
	"""Tokenizes text, and fits to a KMeans model"""

	def __init__(self):
		self.stemmer = PorterStemmer()
		self.vectorizer = TfidfVectorizer()
		self.clf = KMeans(10)

	def tokenize(self,title):

		title = title.decode('latin1')
		title = [word for word in title.lower().split() if word not in punctuation]
		title = [self.stemmer.stem(word) for word in title]
		return " ".join(title)

	def fit(self,text):
		features = np.array([self.tokenize(title) for title in text])
	    
		X = self.vectorizer.fit_transform(features).toarray()
		self.clf.fit(X)

		return self.clf.predict(X)

	def predict_one(self,line):
		query = self.tokenize(line)
		query_vector = self.vectorizer.transform([query]).toarray()
		return self.clf.predict(query_vector)[0]



if '__name__' == '__main__':
	"""Fits all the titles from the dresses in the dataset
	to a KMeans model, and pickles the clustering model for use
	in the app"""

	
	clusterer = clusters.TextClusters()

	PASSWORD = '------------'
	conn = psycopg2.connect("dbname='postgres' user='Tyler' host='zipfianproject.co9ijljkg9gd.us-east-1.rds.amazonaws.com' port='5432' password=PASSWORD")
	cur = conn.cursor()

	cur.execute('''SELECT id, title from clothes;''')
	ids, titles = zip(*cur.fetchall())

	cur.execute('''SELECT id, title from clothes;''')
	ids, titles = zip(*cur.fetchall())

	labels = clusterer.fit(titles)

	#Adding the cluster labels to the database
	for dress_id, label in zip(ids, labels):
	    cur.execute('''UPDATE clothes 
	                    SET cluster = %s
	                    WHERE id = %s;''', (int(label), dress_id))
	conn.commit()

	labels = clusterer.fit(titles)
	#Pickle the model for use in the app.
	pickle.dump(clusterer,open('cluster.plk','wb'))

