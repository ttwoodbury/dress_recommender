import cPickle as pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

class FeatureScaler:
	def __init__(self):
		self.color_scaler = pickle.load(open('transformers/color_scaler.plk','rb'))
		self.top_scaler = pickle.load(open('transformers/top_scaler.plk','rb'))
		self.color_pca = pickle.load(open('transformers/color_pca.plk','rb'))

	def scale_color(self,feature):
		scaled = self.color_scaler.transform(feature)
		scaled_pca = self.color_pca.transform(scaled)
		return scaled_pca

	def scale_top(self, feature):
		return self.top_scaler.transform(feature)