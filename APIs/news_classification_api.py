"""
To run this app, in your terminal:
> python news_classification_api.py

Navigate to:
> http://localhost:8080/ui/
"""
import connexion
from joblib import load
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                        stop_words='english')

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='')
application = app.app

# Load our pre-trained model
clf = load('./Classificatin.joblib')

# Implement a simple health check function (GET)
def health():
    # Test to make sure our service is actually healthy
    try:
        predict('Hello I am a sports article!')
    except:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}

# Implement our predict function
def predict(article):
    # Accept the feature values provided as part of our POST
    # Use these as input to clf.predict()

    article = [article]
    features = tfidf.fit_transform(article).toarray()
    prediction = clf.predict(features)

    print(prediction)

    # Map the predicted value to an actual class

    # Return the prediction as a json
    return {"prediction" : prediction}

# Read the API definition for our service from the yaml file
app.add_api("news_classification_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
