from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD


""" Normalizes all the image features, and loads them into a database.  Pickles
the scaling and PCA objects to use in the app.  (They are incorperated into the FeatureScaler class found in the
	scaler.py file""")

	
PASSWORD = '-------------'
engine = create_engine("postgresql://Tyler:%s@zipfianproject.co9ijljkg9gd.us-east-1.rds.amazonaws.com:5432/postgres")%PASSWORD
conn = psycopg2.connect("dbname='postgres' user='Tyler' host='zipfianproject.co9ijljkg9gd.us-east-1.rds.amazonaws.com' port='5432' password=%s")%PASSWORD
cur = conn.cursor()

image_data = pd.read_pickle('image_features.plk')
image_data['top_colors'] = image_date['top_colors'].apply(lambda x: list(x))
image_data['top_colors'] = image_date['top_colors'].apply(lambda x: map(lambda y: int(y),x))

scalar = StandardScaler()

df = image_data.copy()
stored_features_col = np.array(df['color_features'].values.tolist())
color_scaled = scalar.fit_transform(stored_features_col)

stored_features_top = np.array(df['top_colors'].values.tolist())
scalar_top = StandardScaler()
top_scaled = scalar.fit_transform(stored_features_top)

spca = TruncatedSVD(50)
color_pca = spca.fit_transform(color_scaled)

gl_or = np.array(df.gl_or.values.tolist())

data = pd.DataFrame({'color_scaled': color_scaled.tolist(), 'color_pca': color_pca.tolist(), 
			'gl_or': gl_or, 'top_scaled': top_scaled.tolist(), 'img_id':img_ids.tolist()})

#load data into the database
data.load('features_scaled2', engine, index = False, if_exists = 'append')