"""
To run this app, in your terminal:
> python news_classification_api.py

Navigate to:
> http://localhost:8080/ui/
"""
import connexion
from joblib import load
import pickle

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='')
application = app.app

# Load our pre-trained model
clf = load('Classification.joblib')

#load our pre-trained vectorizer
tfidf = pickle.load(open("tfidf.pickle", "rb"))

#Dictionary with all our mapped categories
category_dict = {
        0: "World News",
        1: "Media",
        2: "Black Voices",
        3: "Entertainment",
        4: "Crime",
        5: "Comedy",
        6: "Politics",
        7: "Women",
        8: "Queer Voices",
        9: "Latino Voices",
        10: "Religion",
        11: "Education",
        12: "Science",
        13: "Tech",
        14: "Business",
        15: "Sport",
        16: "Travel",
        17: "Impact"
    }

# Implement a simple health check function (GET)
def health():
    # Test to make sure our service is actually healthy
    try:
        features = tfidf.transform(['Hello I am a sports article!'])
        clf.predict(features)
    except:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}

# Implement our predict function
def predict(article):
    # Accept the feature values provided as part of our POST
    # Use these as input to clf.predict()

    features = tfidf.transform([article])
    prediction = clf.predict(features)
    prediction = category_dict.get(prediction[0])

    # Return the prediction as a json
    return {"prediction" : prediction}

# Read the API definition for our service from the yaml file
app.add_api("news_classification_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
