"""
To run this app, in your terminal:
> python diabetes_classification_api.py

Navigate to:
> http://localhost:8080/ui/
"""
import connexion
from joblib import load

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='')
application = app.app

# Load our pre-trained model
clf = load('./isidora-attempt3.joblib')

# Implement a simple health check function (GET)
def health():
    # Test to make sure our service is actually healthy
    try:
        predict(4, 183, 0, 0, 0, 28.4, 0.212, 36)
    except:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}

# Implement our predict function
def predict(pregnancies, glucose3, blood_press, skin_thick, insulin, bmi, diab_func, age):
    # Accept the feature values provided as part of our POST
    # Use these as input to clf.predict()
    prediction = clf.predict([[pregnancies, glucose3, blood_press, skin_thick, insulin, bmi, diab_func, age]])

    # Map the predicted value to an actual class
    if prediction[0] == 0:
        predicted_class = "Non-diabetic"
    elif prediction[0] == 1:
        predicted_class = "Diabetic"
    else:
        predicted_class = "We don't know"

    # Return the prediction as a json
    return {"prediction" : predicted_class}

# Read the API definition for our service from the yaml file
app.add_api("diabetes_classification_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
