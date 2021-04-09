"""
To run this app, in your terminal:
> python news_classification_api.py

Navigate to:
> http://localhost:8080/ui/
"""
import connexion
from joblib import load

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='')
application = app.app

# Load our pre-trained model
clf = load('./Classification_joblib')

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
    prediction = clf.predict([[article]])

    # Map the predicted value to an actual class

    # Return the prediction as a json
    return {"prediction" : prediction}

# Read the API definition for our service from the yaml file
app.add_api("news_classification_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
