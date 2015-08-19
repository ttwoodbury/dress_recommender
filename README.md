#Introduction / Motivation

I made a web app that allows you to put in a URL of a dress, and it will return the top three matches from clotheing websites that I scaped.  The idea is to put in a link to an expensive peice of clothing and the app will return a cheaper alternative.


##Outline

- Scape images and metadata from Forever21, and Jollychic.  I have 5000 unique dresses and 16000 images

- Process them images by removing faces, and backgrounds.

- Featurize by extracting top color from the images, and creating histograms of the colors

- Preform Tf-Idf on the titles of the dresses, and do KMeans to cluster them

- Deploy an app that allows users to filter images by dresses by price and description,
and return most similar images

##Scoring Metric
This was an unsupervied problem which is inherently hard to evaluate.  I came up with a scoring metric that measures how the model does at predicting images of the same dress as being similar.  For every image in my dataset, I would calculate the fraction of 10 most similar images that were of the same dress. The final score was the average of these fractions. 

This metric allowed me to explore what features worked best.  I also used it to determine how to weight the features, and how many components to use when I ran SVD to reduce the dimentiallity of my features.


#Packages Used
- Scikit-learn: StandardScaler to normalize the features, TruncatedSVD to reduce the dimensionality of color histograms, and KMeans for text clustering

- Web Scaping Tools: BeautifulSoup, Tor, urllib2

- Scikit-image, Pillow, and OpenCV for image processing.

- Psycopg2, and PostgresSQL to store the data

- AWS: S3, EC2, and and RDS


