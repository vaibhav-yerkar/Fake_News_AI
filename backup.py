
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from flask import Flask, request, jsonify
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Initialize Flask app
app = Flask(__name__)

# Constants
MAX_FEATURES = 10000
MAX_TITLE_LENGTH = 50
MAX_AUTHOR_LENGTH = 20
MAX_TEXT_LENGTH = 500
EMBEDDING_DIM = 100

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Tokenize
        words = text.split()
        # Remove stopwords and stem
        words = [stemmer.stem(word) for word in words if word not in stop_words]
        return ' '.join(words)
    return ''

# Load and preprocess data
def load_data():
    # This would normally load from a file, but we're using the sample data from the prompt
    data = pd.read_csv('./train.csv')
    
    # In a real-world scenario, you'd have more data:
    # data = pd.read_csv('fake_news_dataset.csv')
    
    # Preprocess title, author, and text columns
    data['title_proc'] = data['title'].apply(preprocess_text)
    data['author_proc'] = data['author'].apply(preprocess_text)
    data['text_proc'] = data['text'].apply(preprocess_text)
    
    return data

# Build and train model
def build_model(data):
    # Tokenize title
    title_tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token="<OOV>")
    title_tokenizer.fit_on_texts(data['title_proc'])
    title_sequences = title_tokenizer.texts_to_sequences(data['title_proc'])
    title_padded = pad_sequences(title_sequences, maxlen=MAX_TITLE_LENGTH, padding='post')
    
    # Tokenize author
    author_tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token="<OOV>")
    author_tokenizer.fit_on_texts(data['author_proc'])
    author_sequences = author_tokenizer.texts_to_sequences(data['author_proc'])
    author_padded = pad_sequences(author_sequences, maxlen=MAX_AUTHOR_LENGTH, padding='post')
    
    # Tokenize text
    text_tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token="<OOV>")
    text_tokenizer.fit_on_texts(data['text_proc'])
    text_sequences = text_tokenizer.texts_to_sequences(data['text_proc'])
    text_padded = pad_sequences(text_sequences, maxlen=MAX_TEXT_LENGTH, padding='post')
    
    # Save tokenizers for inference
    with open('title_tokenizer.pickle', 'wb') as handle:
        pickle.dump(title_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('author_tokenizer.pickle', 'wb') as handle:
        pickle.dump(author_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('text_tokenizer.pickle', 'wb') as handle:
        pickle.dump(text_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Define model architecture
    # Title input
    title_input = Input(shape=(MAX_TITLE_LENGTH,), name='title_input')
    title_embedding = Embedding(MAX_FEATURES, EMBEDDING_DIM, name='title_embedding')(title_input)
    title_lstm = LSTM(64, return_sequences=True, name='title_lstm')(title_embedding)
    title_global_pool = GlobalMaxPooling1D(name='title_global_pool')(title_lstm)
    
    # Author input
    author_input = Input(shape=(MAX_AUTHOR_LENGTH,), name='author_input')
    author_embedding = Embedding(MAX_FEATURES, EMBEDDING_DIM, name='author_embedding')(author_input)
    author_lstm = LSTM(32, return_sequences=True, name='author_lstm')(author_embedding)
    author_global_pool = GlobalMaxPooling1D(name='author_global_pool')(author_lstm)
    
    # Text input
    text_input = Input(shape=(MAX_TEXT_LENGTH,), name='text_input')
    text_embedding = Embedding(MAX_FEATURES, EMBEDDING_DIM, name='text_embedding')(text_input)
    text_lstm = LSTM(128, return_sequences=True, name='text_lstm')(text_embedding)
    text_global_pool = GlobalMaxPooling1D(name='text_global_pool')(text_lstm)
    
    # Combine features
    concatenated = Concatenate()([title_global_pool, author_global_pool, text_global_pool])
    dense1 = Dense(64, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense1)
    
    # Create and compile model
    model = Model(inputs=[title_input, author_input, text_input], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # In a real scenario, we would split the data into train/test sets
    # For this example with minimal data, we'll just fit on the entire dataset
    model.fit(
        [title_padded, author_padded, text_padded], 
        data['label'], 
        epochs=10,  # Reduced epochs for demonstration
        batch_size=32,
        validation_split=0.2
    )
    
    # Save model
    model.save('fake_news_model.h5')
    
    return model, title_tokenizer, author_tokenizer, text_tokenizer

# Load model and tokenizers
def load_model_and_tokenizers():
    try:
        # Load saved model)
        model = tf.keras.models.load_model('fake_news_model.h5')
        
        # Load tokenizers
        with open('title_tokenizer.pickle', 'rb') as handle:
            title_tokenizer = pickle.load(handle)
        with open('author_tokenizer.pickle', 'rb') as handle:
            author_tokenizer = pickle.load(handle)
        with open('text_tokenizer.pickle', 'rb') as handle:
            text_tokenizer = pickle.load(handle)
            
        return model, title_tokenizer, author_tokenizer, text_tokenizer
    except:
        print("Model training required. Training a new model...")
        data = load_data()
        return build_model(data)

# Prediction function
def predict_fake_news(title, author, text, model, title_tokenizer, author_tokenizer, text_tokenizer):
    # Preprocess inputs
    title_proc = preprocess_text(title)
    author_proc = preprocess_text(author)
    text_proc = preprocess_text(text)
    
    # Tokenize and pad
    title_seq = title_tokenizer.texts_to_sequences([title_proc])
    title_pad = pad_sequences(title_seq, maxlen=MAX_TITLE_LENGTH, padding='post')
    
    author_seq = author_tokenizer.texts_to_sequences([author_proc])
    author_pad = pad_sequences(author_seq, maxlen=MAX_AUTHOR_LENGTH, padding='post')
    
    text_seq = text_tokenizer.texts_to_sequences([text_proc])
    text_pad = pad_sequences(text_seq, maxlen=MAX_TEXT_LENGTH, padding='post')
    
    # Make prediction
    prediction = model.predict([title_pad, author_pad, text_pad])[0][0]
    
    return float(prediction)

model, title_tokenizer, author_tokenizer, text_tokenizer = load_model_and_tokenizers()

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Server is running!'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        title = data.get('title', '')
        author = data.get('author', '')
        text = data.get('text', '')
        
        prediction = predict_fake_news(title, author, text, model, title_tokenizer, author_tokenizer, text_tokenizer)
        
        is_fake = prediction >= 0.5
        
        response = {
            'prediction': float(prediction),
            'is_fake': bool(is_fake),
            'message': 'This news appears to be fake.' if is_fake else 'This news appears to be real.'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Main function to run the app
if __name__ == '__main__':
    # Preload the model
    print("Loading model and tokenizers...")
    model, title_tokenizer, author_tokenizer, text_tokenizer = load_model_and_tokenizers()
    print("Model loaded successfully!")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=8008)