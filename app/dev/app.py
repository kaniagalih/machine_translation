import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st 
from pathlib import Path
import pickle

# Attention
# https://colab.research.google.com/drive/1XrjPL3O_szhahYZW0z9yhCl9qvIcJJYW

import tensorflow as tf
from tensorflow.keras.layers import Concatenate,Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Bidirectional, LSTM

import tensorflow as tf
tf.compat.v1.logging.set_verbosity

class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                print('wa.s>',W_a_dot_s.shape)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>',U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim
            return fake_state

        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
        fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]

def create_inference_models(model, num_encoder_tokens, num_decoder_tokens, max_length_src, hidden_units, use_attention):
    # Print out diagnostic information
    print("Model input shape:", model.input)
    print("Model layers:", [layer.name for layer in model.layers])
    
    # Print out shapes of key layers
    for layer in model.layers:
        try:
            print(f"Layer {layer.name} output shape: {layer.output_shape}")
        except:
            print(f"Could not get output shape for layer {layer.name}")

    # extract layers with more robust error handling
    try:
        embedding_layers = [i for i in model.layers if "embedding" in i.name]
        bidirectional_layer = [i for i in model.layers if "bidirectional" in i.name][0]
        lstm_layer = [i for i in model.layers if "lstm" in i.name][0]
        concat_layers = [i for i in model.layers if "concat" in i.name]
        dense_layer = [i for i in model.layers if "dense" in i.name][0]
    except IndexError as e:
        print(f"Error finding layers: {e}")
        raise

    # Modify to handle different model input configurations
    encoder_inputs = model.input[0] if isinstance(model.input, list) else model.input  # Encoder input
    
    # More robust handling of bidirectional layer output
    if isinstance(bidirectional_layer.output, list):
        # If output is a list, assume it contains [outputs, forward_h, forward_c, backward_h, backward_c]
        encoder_outputs = bidirectional_layer.output[0]
        forward_h = bidirectional_layer.output[1]
        forward_c = bidirectional_layer.output[2]
        backward_h = bidirectional_layer.output[3]
        backward_c = bidirectional_layer.output[4]
    else:
        # Fallback: use the entire bidirectional output
        encoder_outputs = bidirectional_layer.output
        forward_h = backward_h = forward_c = backward_c = None

    # Ensure consistent dimensions for states
    state_h = Concatenate()([forward_h, backward_h]) if forward_h is not None and backward_h is not None else None
    state_c = Concatenate()([forward_c, backward_c]) if forward_c is not None and backward_c is not None else None

    # Fallback for encoder states
    encoder_states = [state_h, state_c] if state_h is not None and state_c is not None else []

    # Create encoder model with more flexible output
    encoder_model = Model(encoder_inputs, 
                          [encoder_outputs] + encoder_states if encoder_states else [encoder_outputs])

    # Print out encoder model output shapes for debugging
    print("Encoder model output shapes:", [output.shape for output in encoder_model.outputs])

    # Decoder inference model
    decoder_inputs = model.input[1] if isinstance(model.input, list) else model.input  # Decoder input
    
    # Determine the correct hidden units size
    print(f"Hidden units: {hidden_units}")
    print(f"Encoder outputs shape: {encoder_outputs.shape}")

    # More robust input shape handling
    # state_shape = (hidden_units * 2,) if hidden_units is not None else encoder_outputs.shape[-1]
    state_shape = (hidden_units,)
    
    decoder_state_input_h = Input(shape=state_shape)
    decoder_state_input_c = Input(shape=state_shape)
    
    # Modify to handle different max length scenarios
    decoder_hidden_state_input = Input(shape=(max_length_src, encoder_outputs.shape[-1]))

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    dec_emb = embedding_layers[1](decoder_inputs)  # Decoder embedding layer
    
    decoder_lstm = lstm_layer
    decoder_outputs, state_h, state_c = decoder_lstm(dec_emb, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    if use_attention:
        try:
            attention_layer = model.get_layer("attention_layer")
            attention_result, _ = attention_layer([decoder_hidden_state_input, decoder_outputs])
            decoder_concat_input = concat_layers[-1]([decoder_outputs, attention_result])
            decoder_outputs = dense_layer(decoder_concat_input)
        except Exception as e:
            print(f"Attention layer error: {e}")
            decoder_outputs = dense_layer(decoder_outputs)
    else:
        decoder_outputs = dense_layer(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs, decoder_hidden_state_input] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return encoder_model, decoder_model

def decode_sequence(input_seq, encoder_model, decoder_model, max_length_target, target_token_index, reverse_target_token_index):
    # Encode the input sequence to get the encoder state
    encoder_outputs, state_h, state_c = encoder_model.predict(input_seq)

    # Find the start token dynamically
    start_tokens = [token for token, index in target_token_index.items() if 'start' in token.lower()]
    
    if not start_tokens:
        raise KeyError("No start token found in target_token_index. Please check your tokenizer.")
    
    start_token = start_tokens[0]  # Take the first start token found

    # Prepare the target sequence (start with the start token)
    target_seq = np.zeros((1, 1))  # Shape (1, 1) for the first input to the decoder
    target_seq[0, 0] = target_token_index[start_token]

    # Initialize the output string
    decoded_sentence = ''

    # Loop for the maximum length of the target sequence
    for _ in range(max_length_target):
        # Predict the next token
        output_tokens, h, c = decoder_model.predict(
            [target_seq, encoder_outputs, state_h, state_c],  # Ensure the shapes match
            verbose=0
        )

        # Get the predicted token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        # Find end tokens dynamically
        end_tokens = [token for token, index in target_token_index.items() if 'end' in token.lower()]
        
        # Check if the sampled token is the end token
        if end_tokens and sampled_token_index == target_token_index[end_tokens[0]]:
            break

        # Get the sampled character
        if sampled_token_index in reverse_target_token_index:
            sampled_char = reverse_target_token_index[sampled_token_index]
        else:
            # Handle the case where the key is not found
            sampled_char = ""  # or any other appropriate handling
            # print(f"Warning: Token index {sampled_token_index} not found in reverse_target_token_index")

        # Append the predicted character to the output sentence
        decoded_sentence += sampled_char

        # Update the target sequence (for the next time step)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update the states
        state_h, state_c = h, c

    return decoded_sentence

def preprocess_text(text, language='java'):
    """
    Preprocess the input text
    :param text: Input text to preprocess
    :param language: 'java' for Javanese, 'indo' for Indonesian
    :return: Preprocessed text
    """
    text = text.lower()  # Convert to lowercase
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text

def translate(input_text, model_path, tokenizer_path, max_length=35):
    """
    Translate Javanese text to Indonesian
    
    :param input_text: Javanese text to translate
    :param model_path: Path to the saved model file
    :param tokenizer_path: Path to the tokenizer pickle file
    :param max_length: Maximum length of translation
    :return: Translated Indonesian text
    """
    # Load tokenizer information
    # tokenizer_path = Path("D:/SKRIPSI/machine_translation/model/tokenizer_info_(4).pkl'")

    with open(str(tokenizer_path), 'rb') as file:
        tokenizer_info = pickle.load(file)
        

    
    # Get token indices from tokenizer info
    input_token_index = tokenizer_info['input_token_index']
    target_token_index = tokenizer_info['target_token_index']
    
    # Check the target token index
    print("Target token index:", target_token_index)
    
    # Load the model with custom objects
    model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
    
    # Preprocess input text
    preprocessed_text = preprocess_text(input_text, 'java')
    
    # Get token indices from tokenizer info
    input_token_index = tokenizer_info['input_token_index']
    target_token_index = tokenizer_info['target_token_index']
    
    # Create reverse token indices
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((index, char) for char, index in target_token_index.items())
    
    # Tokenize input text
    input_seq = np.zeros((1, model.input[0].shape[1]), dtype='int32')
    for t, char in enumerate(preprocessed_text.split()):
        if t < model.input[0].shape[1]:
            if char in input_token_index:
                input_seq[0, t] = input_token_index[char]
    
    # Determine the correct hidden units size
    hidden_units = 64  # Set this to the number of hidden units you used in the model

    # If using Bidirectional LSTM, the output will be double the hidden units
    if 'bidirectional' in [layer.name for layer in model.layers]:
        hidden_units *= 2  # Ensure hidden_units is doubled

    # More robust input shape handling
    state_shape = (hidden_units,)  # This should be (hidden_units * 2,) if using Bidirectional LSTM

    use_attention = 'attention_layer' in [layer.name for layer in model.layers]
    num_encoder_tokens = len(input_token_index)
    num_decoder_tokens = len(target_token_index)
    
    print(f"Hidden units: {hidden_units}")
    print(f"Use Attention: {use_attention}")

    encoder_model, decoder_model = create_inference_models(
        model, 
        num_encoder_tokens, 
        num_decoder_tokens, 
        model.input[0].shape[1], 
        hidden_units,  # Pass the correct hidden units to the inference model
        use_attention
    )
    
    # Perform translation
    translated_text = decode_sequence(
        input_seq, 
        encoder_model, 
        decoder_model, 
        max_length, 
        target_token_index, 
        reverse_target_char_index
    )
    
    return translated_text

# Streamlit app
st.title("Translation App")

# Default paths
DEFAULT_MODEL_PATH = '../model/Bi-LSTM-Attention-64-Dropout_0.2-Fold_4.keras'
# DEFAULT_TOKENIZER_PATH = Path('D:/SKRIPSI/machine_translation/model/tokenizer_info_(4).pkl')
from pathlib import WindowsPath
DEFAULT_TOKENIZER_PATH = WindowsPath('D:/SKRIPSI/machine_translation/model/tokenizer_info_(4).pkl')


# Input text from user
input_text = st.text_area("Enter the input text:")

# Translate button
if st.button("Translate"):
    if input_text.strip():
        try:
            translated_text = translate(input_text, DEFAULT_MODEL_PATH, DEFAULT_TOKENIZER_PATH)
            st.success(f"Translated Text: {translated_text}")
        except Exception as e:
            st.error(f"Error occurred: {e}")
    else:
        st.warning("Please enter some text to translate.")