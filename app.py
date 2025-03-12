from flask import Flask, render_template, request, session
from flask_socketio import SocketIO, emit
import json
import numpy as np
import random
import time
import tensorflow as tf
import re
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the Keras model
model = tf.keras.models.load_model('disease_prediction_model.keras')

# Load the disease mapping
with open('mapping.json', 'r') as f:
    disease_mapping = json.load(f)

# Create reverse mapping (index to disease)
label_to_disease = {v: k for k, v in disease_mapping.items()}

# Load the vectorizer - we need to use the same vectorizer that was used during training
try:
    # Try to load the saved vectorizer
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Loaded vectorizer successfully!")
except:
    print("Warning: Vectorizer not found. Creating a new TfidfVectorizer with max_features=5000.")
    # Create a new vectorizer with the same parameters as in training
    vectorizer = TfidfVectorizer(max_features=5000)
    # Note: This new vectorizer won't have the same vocabulary as the one used in training,
    # so it's better to save and load the original vectorizer

# Preprocessing and prediction functions
def preprocess_symptoms(symptoms):
    """
    Preprocess the input symptoms to match the training data format.
    """
    # Convert to lowercase
    symptoms = symptoms.lower()
    # Remove punctuation and special characters
    symptoms = re.sub(r"[^\w\s]", "", symptoms)
    # Remove extra spaces
    symptoms = " ".join(symptoms.split())
    return symptoms

def predict_diseases(symptoms, top_n=5, threshold=5):
    """
    Predict diseases based on input symptoms.

    Args:
        symptoms (str): Input symptoms as a string.
        top_n (int): Number of top predictions to return.
        threshold (float): Minimum probability threshold for predictions (in percentage).

    Returns:
        dict: A dictionary of predicted diseases and their probabilities.
    """
    try:
        # Preprocess the input symptoms
        symptoms = preprocess_symptoms(symptoms)
        
        # Transform the symptoms using the same vectorizer used during training
        # This ensures the feature space is the same
        symptoms_vectorized = vectorizer.transform([symptoms]).toarray()
        
        # Check the shape to ensure it matches what the model expects
        print(f"Input shape: {symptoms_vectorized.shape}")
        
        # If the shape doesn't match what the model expects, we need to pad or truncate
        expected_features = 5000  # Based on max_features in the vectorizer
        actual_features = symptoms_vectorized.shape[1]
        
        if actual_features < expected_features:
            # Pad with zeros to match the expected number of features
            padding = np.zeros((symptoms_vectorized.shape[0], expected_features - actual_features))
            symptoms_vectorized = np.hstack((symptoms_vectorized, padding))
            print(f"Padded input shape: {symptoms_vectorized.shape}")
        elif actual_features > expected_features:
            # Truncate to match the expected number of features
            symptoms_vectorized = symptoms_vectorized[:, :expected_features]
            print(f"Truncated input shape: {symptoms_vectorized.shape}")

        # Make predictions
        predictions = model.predict(symptoms_vectorized)[0]

        # Map predictions to disease names and percentages
        disease_probabilities = {label_to_disease[i]: float(predictions[i]) * 100 for i in range(len(predictions))}

        # Filter diseases with probability >= threshold
        filtered_diseases = {k: v for k, v in disease_probabilities.items() if v >= threshold}

        # Sort by probability (descending) and return top N predictions
        sorted_diseases = dict(sorted(filtered_diseases.items(), key=lambda item: item[1], reverse=True)[:top_n])

        return sorted_diseases
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return {}

# Model statistics tracker
class ModelStats:
    def __init__(self):
        self.prediction_history = []
        self.total_predictions = 0
        self.avg_confidence = 0
        self.avg_processing_time = 0
        
    def get_stats(self):
        # Get the most recent predictions (up to 10)
        recent_history = self.prediction_history[-10:] if self.prediction_history else []
        print(f"Sending prediction history: {recent_history}")
        
        return {
            "total_predictions": self.total_predictions,
            "avg_confidence": round(self.avg_confidence, 4) if self.total_predictions > 0 else 0,
            "avg_processing_time": round(self.avg_processing_time, 4) if self.total_predictions > 0 else 0,
            "prediction_history": recent_history
        }
        
    def process_message(self, message):
        start_time = time.time()
        
        # Process the message with the actual model
        predictions = predict_diseases(message, top_n=3, threshold=5)
        
        processing_time = time.time() - start_time
        
        # Update statistics
        if predictions:
            # Get the confidence of the top prediction (already in percentage form)
            top_confidence = list(predictions.values())[0] / 100
            
            # Add to prediction history (ensure it's a value between 0 and 1)
            self.prediction_history.append(top_confidence)
            print(f"Added prediction to history: {top_confidence}, history length: {len(self.prediction_history)}")
            
            # Update running averages
            self.total_predictions += 1
            self.avg_confidence = ((self.avg_confidence * (self.total_predictions - 1)) + top_confidence) / self.total_predictions
            self.avg_processing_time = ((self.avg_processing_time * (self.total_predictions - 1)) + processing_time) / self.total_predictions
        else:
            # Even if no predictions meet the threshold, add a zero confidence to show activity in the chart
            self.prediction_history.append(0.0)
            print("No predictions above threshold, added 0.0 to history")
        
        return {
            "predictions": predictions,
            "confidence": round(list(predictions.values())[0] / 100, 4) if predictions else 0,
            "processing_time": round(processing_time, 2),
            "tokens_processed": len(message.split())
        }

model_stats = ModelStats()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('model_stats', model_stats.get_stats())

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('message')
def handle_message(data):
    print('Received message:', data)
    
    # Process the message with our actual model
    message_stats = model_stats.process_message(data['message'])
    
    # Get updated model stats
    model_stats_data = model_stats.get_stats()
    
    # Broadcast the message to all clients except the sender
    emit('message', {
        'user': data['user'],
        'message': data['message'],
        'timestamp': time.strftime('%H:%M:%S')
    }, broadcast=True, include_self=False)
    
    # Send message stats back to the sender
    emit('message_processed', {
        'message': data['message'],
        'stats': message_stats
    })
    
    # Broadcast updated model stats to all clients
    emit('model_stats', model_stats_data, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

