from flask import Flask, request, jsonify
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')

# Initialize the Flask app
app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function (same as the one used during training)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+http\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'ð','',text)
    tweet_tokens = word_tokenize(text)
    filtered_tweets = [w for w in tweet_tokens if not w in stop_words]
    return " ".join(filtered_tweets)

# Define the class labels (adjust as per your dataset's labels)
class_labels = {0: 'Non-Hate', 1: 'Hate'}

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json(force=True)
    
    # Get the text to be classified
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Vectorize the processed text
    text_vector = vectorizer.transform([processed_text])
    
    # Predict the class
    prediction = model.predict(text_vector)
    
    # Return the prediction as a JSON response
    return jsonify({
        'text': text,
        'predicted_class': class_labels[prediction[0]]
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
