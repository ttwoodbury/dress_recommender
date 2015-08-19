from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.decomposition import TruncatedSVD
from sqlalchemy import create_engine
import cPickle as pickle
import pandas as pd



def score_all(image_data, features, weights, dist, pcas = [-1], topk = -1):
    n= len(features)
    if pcas[0] <0:
        pcas = [-1]*n
    
    df = image_data.copy()
    
    img_ids = df['id'].values
    df.set_index('id', inplace = True)
    
    similar_dict = df.groupby('dress_ids').groups
    
    df['similar_images'] = df.dress_ids.apply(lambda x: similar_dict[x])
    similar_imgs = df['similar_images'].values
    
    distances = 0 
    for feature, weight, pca in zip(features,weights,pcas):
    
        stored_features = np.array(df[feature].values.tolist())
    
        if pca > 0:
            spca = TruncatedSVD(pca)
            stored_features = spca.fit_transform(stored_features)

        features_dist = squareform(pdist(stored_features, metric=dist))
        features_dist = np.array(features_dist)
        
        distances+=features_dist*weight
        
    image_scores = []

    for d, sim_imgs in zip(distances,similar_imgs):
        num_imgs = len(sim_imgs)
    
        k = max(topk, num_imgs)
            
        mask = np.argsort(d)
        top_imgs = img_ids[mask][:k]

        image_scores.append(len(set(top_imgs)&set(sim_imgs))*1./num_imgs)

    score = sum(image_scores)*1./len(image_scores)

    return score


if '__name__' == '__main__':
    with open('~/Documents/Zipfian/project-app/.db_password.txt', 'rb') as f:
        PASSWORD = f.readline()
    conn = psycopg2.connect("dbname='postgres' user='Tyler' host='zipfianproject.co9ijljkg9gd.us-east-1.rds.amazonaws.com' port='5432' password=%s")%PASSWORD
    cur = conn.cursor()

    query = '''SELECT 
              images.id as id, 
              clothes.id as dress_ids,  
              color_scaled,
              gl_or,  
              top_scaled,
              color_pca
            FROM 
                images JOIN clothes 
                ON images.dress_id = clothes.id
            JOIN features_scaled2
                ON images.id = features_scaled2.img_id;'''

    image_data = pd.read_sql(query, conn)

    #Run the scoring function on a subset of the data.
    subset_mask = np.random.choice(np.unique(image_data.dress_ids.values),2000, replace = False)
    image_test = image_data[image_data.dress_ids.isin(subset_mask)]

    # Go Through a multiple parameters to test which are the best.
    features = ['color_scaled', 'top_scaled']
    probs = np.arange(1,4,1)/5.
    var = [5,20,50,100,-1]

    for pca in var:
        for p in probs:
            weights = [p,1-p]
            pcas = [pca, -1]
            
            print 'PCA: ', pca
            print 'Weights: ', p
            print score_all(image_test, features, weights, dist = 'euclidean', pcas = pcas, topk = 20)
            print '-----------------------------------------'

    #Find the score for the original GL features
    print score_all(image_test, ['gl_or', [1], dist = 'euclidean', pcas = [-1], topk = 20)
