from flask import Flask
from flask import render_template, url_for
from flask_bootstrap import Bootstrap
import pandas as pd
from flask import request, redirect
import cPickle as pickle
import requests
from scaler import FeatureScaler
import sys
import os

from app_pipeline import extract_all_features
from recommender import Recommender
import psycopg2
app = Flask(__name__)
#from functions import recommender 

@app.route('/')
def hello():
    return render_template("index.html")
                        


@app.route('/predict', methods=['POST', 'GET'] )
def predict():

    if request.method == 'GET':
        return redirect(url_for('hello'))

    # get data from request form, the key is the name you set in your form
    url = request.form['url']
    if not url:
        return render_template("error.html")

    url = str(url)

    extensions = ['jpg','png','jpeg']
    image = False
    for ext in extensions:
        if ext in url.split('.')[-1].lower():
            image = True
    if not image:
        return render_template("error.html")


    r = requests.get(url, headers = {'User-agent': 'Mozilla/5.0'})

    if r.status_code != 200:
    	return render_template("error.html")

    img_features = extract_all_features(url, scaler)


    text = request.form['title']
    max_price = float(request.form['max_price'])
    weight = float(request.form['weight'])/100


    if text:
        label = clf.predict_one(text)
    else: 
        label = -1

    if max_price:
        price = int(max_price)
    else:
        price = 1000

    features = ['color_pca', 'top_scaled']
    weights = [1- weight, weight] 

    new_image_data = image_data[image_data['price']<=max_price]
    if label >= 0:
        new_image_data = new_image_data[new_image_data['cluster']==label]

    rec = Recommender(new_image_data)
    valid = rec.predict(img_features, features, weights)

    if not valid:
        return render_template("error.html")

    rec.top_unique(k=3)
    rec_url = rec.similar_img_files
    rec_dress = rec.similar_dress_urls
    rec_title = rec.similar_titles
    rec_price = [format(price,'.2f') for price in rec.similar_price]

    similar = zip(rec_url, rec_title, rec_dress, rec_price)

    return render_template("predict.html", org_url = url, similar = similar)


@app.route('/clothes_error', methods=['POST'] )
def clothes_error():
    return render_template("error.html")

if __name__ == '__main__':
    with open('~/Documents/Zipfian/project-app/.db_password.txt', 'rb') as f:
        PASSWORD = f.readline()
    conn = psycopg2.connect("dbname='postgres' user='Tyler' host='zipfianproject.co9ijljkg9gd.us-east-1.rds.amazonaws.com' port='5432' password=%s")%PASSWORD
    cur = conn.cursor()
    query = '''SELECT 
              images.id as id, 
              clothes.id as dress_ids,
              price,
              cluster, 
              color_pca,  
              top_scaled
            FROM 
                images JOIN clothes 
                ON images.dress_id = clothes.id
            JOIN tmp
                ON images.id = tmp.img_id;'''


    image_data = pd.read_sql(query, conn)
    Bootstrap(app)
    scaler = FeatureScaler()
    clf = pickle.load(open('cluster.plk','rb'))
    app.run(host='0.0.0.0', port=8080, debug=True)


