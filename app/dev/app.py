import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Concatenate
from bpemb import BPEmb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st  # Streamlit for web interface

from Attention import AttentionLayer  # Custom attention layer

# Disable TF logging (Optional, for cleaner outputs)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Initialize BPE for tokenization
bpe = BPEmb(lang="jv", vs=5000)

# Load tokenizer information
DEFAULT_MODEL_PATH = 'model/64-Bahdanau-Dropout-fold-4.keras'
DEFAULT_TOKENIZER_PATH = "model/tokenizer_info (11).pkl"

# Load tokenizer data
with open(DEFAULT_TOKENIZER_PATH, "rb") as f:
    tokenizer_data = pickle.load(f)
    num_encoder_tokens = tokenizer_data['num_encoder_tokens']
    num_decoder_tokens = tokenizer_data['num_decoder_tokens']
    target_token_index = tokenizer_data['target_token_index']
    input_token_index = tokenizer_data['input_token_index']
    max_length = tokenizer_data['max_length']

reverse_target_token_index = {v: k for k, v in target_token_index.items()}

# Load model
model = load_model(DEFAULT_MODEL_PATH, custom_objects={'AttentionLayer': AttentionLayer})

# Create inference models (encoder & decoder)
def create_inference_models(model, num_encoder_tokens, num_decoder_tokens, max_length_src, hidden_units, use_attention):
    # Extract layers for encoder and decoder
    embedding_layers = [layer for layer in model.layers if "embedding" in layer.name]
    bidirectional_layer = [layer for layer in model.layers if "bidirectional" in layer.name][0]
    lstm_layer = [layer for layer in model.layers if "lstm" in layer.name][0]
    concat_layers = [layer for layer in model.layers if "concat" in layer.name]
    dense_layer = [layer for layer in model.layers if "dense" in layer.name][0]
    attention_layers = [layer for layer in model.layers if "attention" in layer.name]

    # Encoder Model
    encoder_inputs = model.input[0]  # Encoder input
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = bidirectional_layer.output
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)

    # Decoder Model
    decoder_inputs = model.input[1]  # Decoder input
    decoder_state_input_h = Input(shape=(hidden_units * 2,))
    decoder_state_input_c = Input(shape=(hidden_units * 2,))
    decoder_hidden_state_input = Input(shape=(max_length_src, hidden_units * 2))

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    dec_emb = embedding_layers[1](decoder_inputs)
    decoder_lstm = lstm_layer
    decoder_outputs, state_h, state_c = decoder_lstm(dec_emb, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    if use_attention:
        attention_layer = attention_layers[-1]
        attention_result, _ = attention_layer([decoder_hidden_state_input, decoder_outputs])
        decoder_concat_input = concat_layers[-1]([decoder_outputs, attention_result])
        decoder_outputs = dense_layer(decoder_concat_input)
    else:
        decoder_outputs = dense_layer(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs, decoder_hidden_state_input] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return encoder_model, decoder_model

# Create encoder and decoder models
encoder_model, decoder_model = create_inference_models(
    model=model,
    num_encoder_tokens=num_encoder_tokens,
    num_decoder_tokens=num_decoder_tokens,
    max_length_src=max_length,
    hidden_units=64,
    use_attention=True
)

# Text preprocessing
def preprocess_text(text):
    """Preprocess input text: lowercasing and removing non-alphanumeric characters."""
    return ''.join([char.lower() if char.isalnum() or char.isspace() else '' for char in text])

# Tokenization
def tokenize(text, token_index):
    """Tokenize input text based on provided token index."""
    return [token_index.get(word, 1) for word in text.split()]

# Translate input text to target language
def translate(input_text):
    """Translate the input text from source language to target language."""
    # Preprocess the text
    preprocessed_text = preprocess_text(input_text)
    
    # Tokenize input text
    input_seq = tokenize(preprocessed_text, input_token_index)
    
    # Pad sequences to the expected length
    padded_seq = pad_sequences([input_seq], maxlen=max_length, padding="post")
    
    # Decode the sequence
    translated_text = decode_sequence(
        padded_seq,
        encoder_model,
        decoder_model,
        max_length,
        target_token_index,
        reverse_target_token_index
    )
    
    return translated_text

def decode_sequence(input_seq, encoder_model, decoder_model, max_length_target, target_token_index, reverse_target_token_index):
    """Decode the input sequence into a translated sentence."""
    # Encode the input as state vectors
    encoder_outputs, state_h, state_c = encoder_model.predict(input_seq, verbose=0)
    states_value = [state_h, state_c]

    # Initialize target sequence with start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_token_index['<start>']

    decoded_sentence = ''
    
    for _ in range(max_length_target):
        output_tokens, h, c = decoder_model.predict([target_seq, encoder_outputs] + states_value, verbose=0)
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_token_index.get(sampled_token_index, "<OOV>")
        
        # Stop condition
        if sampled_char == '<end>':
            break
        
        decoded_sentence += ' ' + sampled_char
        
        # Update target sequence and states
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()

def main():            
    import streamlit as st
    import sqlite3
    from db import create_table, insert_data
    
    # Initialize the database table
    create_table()
    
    # Initialize session state
    st.session_state.setdefault('translations', [])
    st.session_state.setdefault('feedback_submitted', False)
    
    # Application title
    st.title("Javanese Translation App")

    # User input for name and age
    name = st.text_input("Enter your name:")
    age = st.number_input("Enter your age:", min_value=0, max_value=120)

    # Input for text in Javanese
    jawa_text = st.text_area("Enter the input text in javanese:")

    # Button to translate text from Javanese to Indonesian
    if st.button("Translate"):
        if jawa_text and name and age:
            translated_text = translate(jawa_text)
            st.success(f"Translated Text: {translated_text}")
            
            # Store the translation in session state
            st.session_state.translations.append({
                'jawa_text': jawa_text,
                'indonesia_text': translated_text
            })
        else:
            st.error("Please fill out all fields!")

    # Display previous translations
    if st.session_state.translations:
        st.write("Previous Translation:")
        for idx, translation in enumerate(st.session_state.translations):
            st.write(f"{idx + 1}. Javanese Text: {translation['jawa_text']} | Translation: {translation['indonesia_text']}")

    # Feedback section
    if st.button("Give Feedback"):
        if st.session_state.translations:
            st.session_state.feedback_submitted = True
        else:
            st.warning("Please do the translation first.")

    if st.session_state.feedback_submitted:
        # Input for rating (1-5)
        rating = st.slider("Give Rating (1-5):", min_value=1, max_value=5, value=3)
        suggestion = st.text_input("Your Suggestion:")

        if st.button("Send Feedback"):
            for translation in st.session_state.translations:
                insert_data(name, age, translation['jawa_text'], translation['indonesia_text'], rating, suggestion)
            st.success("Feedback successfully saved to database!")
            st.session_state.feedback_submitted = False
            st.session_state.translations = [] 

if __name__ == "__main__":
    main()