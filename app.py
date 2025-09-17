from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

def preprocess_text(text):
    """Preprocess the complaint text"""
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get complaint text from request
        data = request.get_json()
        complaint_text = data.get('complaint', '')
        
        if not complaint_text:
            return jsonify({'error': 'No complaint text provided'}), 400
        
        # Preprocess the text
        cleaned_text = preprocess_text(complaint_text)
        
        # Transform using TF-IDF
        text_vector = tfidf.transform([cleaned_text]).toarray()
        
        # Make prediction
        prediction = model.predict(text_vector)
        probability = model.predict_proba(text_vector)
        
        # Prepare response
        result = {
            'prediction': 'Solved' if prediction[0] == 1 else 'Not Solved',
            'confidence': float(max(probability[0])),
            'probabilities': {
                'solved': float(probability[0][1]),
                'not_solved': float(probability[0][0])
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        complaints = data.get('complaints', [])
        
        if not complaints:
            return jsonify({'error': 'No complaints provided'}), 400
        
        results = []
        for complaint in complaints:
            cleaned_text = preprocess_text(complaint)
            text_vector = tfidf.transform([cleaned_text]).toarray()
            prediction = model.predict(text_vector)
            probability = model.predict_proba(text_vector)
            
            results.append({
                'complaint': complaint,
                'prediction': 'Solved' if prediction[0] == 1 else 'Not Solved',
                'confidence': float(max(probability[0])),
                'probabilities': {
                    'solved': float(probability[0][1]),
                    'not_solved': float(probability[0][0])
                }
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)