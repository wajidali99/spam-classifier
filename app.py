from unittest import result

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from flask import Flask ,render_template,request

# Flask app banao
app = Flask(__name__)

# Model train karo
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[["v1", "v2"]]
data.columns = ["label", "message"]

x = data["message"]
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

cv = CountVectorizer()
x_train_cv = cv.fit_transform(x_train)

model = MultinomialNB()
model.fit(x_train_cv, y_train)

# routes

@app.route("/")
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])

def predict():
    message = request.form.get("message")
    message_cv = cv.transform([message])
    result = model.predict(message_cv)[0]
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

