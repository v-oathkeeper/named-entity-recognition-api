# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, SpatialDropout1D, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import re # Import the regular expression module

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Create a mapping for full tag names ---
TAG_NAMES = {
    'B-geo': 'Geographical Entity',
    'I-geo': 'Geographical Entity',
    'B-gpe': 'Geopolitical Entity',
    'I-gpe': 'Geopolitical Entity',
    'B-per': 'Person',
    'I-per': 'Person',
    'B-org': 'Organization',
    'I-org': 'Organization',
    'B-tim': 'Time indicator',
    'I-tim': 'Time indicator',
    'B-art': 'Artifact',
    'I-art': 'Artifact',
    'B-eve': 'Event',
    'I-eve': 'Event',
    'B-nat': 'Natural Phenomenon',
    'I-nat': 'Natural Phenomenon',
    'O': 'Other'
}


# --- Load Pre-trained Model and Mappings ---
print("INFO: Loading model and mappings...")

# Load the mappings from the JSON file
with open('ner_mappings.json') as f:
    mappings = json.load(f)
    word2idx = mappings['word2idx']
    tag2idx = mappings['tag2idx']
    idx2tag = {int(k): v for k, v in mappings['idx2tag'].items()} # Ensure keys are integers
    MAX_LEN = mappings['max_len']
    WORDS_COUNT = mappings['words_count']
    TAGS_COUNT = mappings['tags_count']

# Re-build the model architecture exactly as it was during training
model = keras.Sequential([
    keras.Input(shape=(MAX_LEN,)),
    Embedding(input_dim=WORDS_COUNT, output_dim=128),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.2)),
    TimeDistributed(Dense(TAGS_COUNT, activation="softmax"))
])

# Load the saved weights
model.load_weights('model_weights.weights.h5')

print("INFO: Model and mappings loaded successfully.")

# --- API and Web Routes ---

@app.route('/')
def home():
    """Renders the main HTML page for the user interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict entities from a sentence."""
    try:
        data = request.json
        sentence = data.get('sentence', '')

        if not sentence:
            return jsonify({'error': 'Sentence is required'}), 400

        # --- IMPROVED TOKENIZATION ---
        # Instead of a simple split, use regex to separate words and punctuation
        words_in_sentence = re.findall(r"[\w']+|[.,!?;]", sentence)
        
        # --- MORE ROBUST PREPROCESSING ---
        # First, try the word as-is. If not found, try its capitalized version.
        # This handles cases where the user types lowercase proper nouns.
        word_indices = [word2idx.get(w, word2idx.get(w.capitalize(), 0)) for w in words_in_sentence]
        padded_sequence = pad_sequences([word_indices], maxlen=MAX_LEN, padding="post", value=WORDS_COUNT - 1)

        # Get model predictions
        predictions = model.predict(padded_sequence)
        predicted_indices = np.argmax(predictions, axis=-1)

        # Format the results
        results = []
        for i, word in enumerate(words_in_sentence):
            if i < len(predicted_indices[0]):
                tag_index = predicted_indices[0][i]
                short_tag = idx2tag.get(tag_index, 'O')
                # Use the new dictionary to get the full, descriptive tag name
                full_tag = TAG_NAMES.get(short_tag, 'Other')
                results.append({'word': word, 'tag': full_tag})

        return jsonify({'results': results})

    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run()

