from flask import Flask, render_template, request, jsonify
import json
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load intents
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Prepare training data
texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern.lower())
        labels.append(intent["tag"])

# TF-IDF (better settings for accuracy)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

# Better ML model (more stable than default)
model = LogisticRegression(max_iter=2000)
model.fit(X, labels)


def get_response(user_input):
    user_input = user_input.lower()

    input_data = vectorizer.transform([user_input])

    probabilities = model.predict_proba(input_data)[0]
    best_index = probabilities.argmax()
    confidence = probabilities[best_index]

    predicted_tag = model.classes_[best_index]

    # Confidence check (VERY IMPORTANT FIX)
    if confidence < 0.4:
        return "Sorry, I didn't understand that. Try asking differently."

    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

    return "Sorry, I don't understand."


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