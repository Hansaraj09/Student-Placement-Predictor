from flask import Flask, jsonify, render_template, request
import pickle
import numpy as np

with open("model.pkl", "rb") as pfile:
    model = pickle.load(pfile)

app = Flask(__name__)


# page 1
@app.route("/")
def home():
    return render_template("index.html")


# page 2
@app.route("/predict", methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    # Make prediction
    prediction = model.predict(final_features)
    output = "Placed" if prediction[0] == 1 else "Not Placed"

    return render_template(
        "index.html", prediction_text="Prediction: {}".format(output)
    )


if __name__ == "__main__":
    app.run(debug=True)
