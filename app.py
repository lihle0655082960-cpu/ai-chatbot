from flask import Flask, render_template, request, jsonify
import json
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Prepare training data
texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern.lower())
        labels.append(intent["tag"])

# IMPROVED TF-IDF (VERY IMPORTANT FIX)
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(texts)

# Stronger model
model = LogisticRegression(max_iter=3000)
model.fit(X, labels)


def get_response(user_input):
    user_input = user_input.lower()

    input_data = vectorizer.transform([user_input])

    prediction = model.predict(input_data)[0]

    # Get matching response
    for intent in data["intents"]:
        if intent["tag"] == prediction:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def chat():
    user_message = request.json["message"]
    response = get_response(user_message)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)