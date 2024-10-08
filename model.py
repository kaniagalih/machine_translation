import yaml
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# Load configuration
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

def masked_accuracy(y_true, y_pred):
    # Reshape y_true if it's 3D
    if len(K.int_shape(y_true)) == 3:
        y_true = tf.squeeze(y_true, axis=-1)

    # Create mask to ignore padding tokens
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))

    # Get accuracy
    accuracies = tf.equal(tf.cast(y_true, dtype=tf.int64),
                          tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.int64))

    # Apply mask
    accuracies = tf.math.logical_and(mask, accuracies)

    # Calculate accuracy
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

def build_model():
    # Encoder
    encoder_inputs = Input(shape=[config['maxlength']])
    enc_emb = Embedding(config['num_encoder_tokens'], config['latent_dim'], mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(config['latent_dim'], return_state=True, return_sequences=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=[config['maxlength']])
    dec_emb_layer = Embedding(config['num_decoder_tokens'], config['latent_dim'], mask_zero=True)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(config['latent_dim'], return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    # Dense output layer
    decoder_dense = Dense(config['indo_vocab_size'], activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Build and return the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    return model

if __name__ == "__main__":
    # Build the model
    model = build_model()

    # Print the model summary
    model.summary()

    # Get optimizer dynamically from TensorFlow
    optimizer = getattr(tf.keras.optimizers, config['optimizer'])()
    
    # Get metrics
    metrics = [masked_accuracy if metric == 'masked_accuracy' else metric for metric in config['metrics']]
    
    # Compile model
    model.compile(optimizer=optimizer, loss=config['loss'], metrics=metrics)
    
    print(f"Model compiled with optimizer: {config['optimizer']}, loss: {config['loss']}")
    print(f"Ready for training with batch size: {config['batch_size']}, epochs: {config['epochs']}")