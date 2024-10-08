import pickle
import numpy as np
import yaml  # Importing the YAML library
from model import build_model
from trainer import Trainer
import tensorflow as tf

# Load configuration from the YAML file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load data from the pickle file
with open("processed_data.pickle", "rb") as f:
    data = pickle.load(f)

# Extract data from the loaded dictionary
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
java_tokenizer = data['java_tokenizer']
indo_tokenizer = data['indo_tokenizer']
java_vocab_size = data['java_vocab_size']
indo_vocab_size = data['indo_vocab_size']
maxlength = data['maxlength']
num_encoder_tokens = data['num_encoder_tokens']
num_decoder_tokens = data['num_decoder_tokens']

# Define model parameters from the config
vocab_size = indo_vocab_size  # Choose based on your task
embedding_dim = 256  # You can also add this to the config if needed
units = config['latent_dim']

# Build the model
model = build_model()

# Initialize Trainer
trainer = Trainer(model)

# Compile the model
trainer.compile(optimizer=getattr(tf.keras.optimizers, config['optimizer'])(), 
                loss=config['loss'], 
                metrics=config['metrics'])

# Fit the model
trainer.fit(x=[X_train, y_train.reshape(y_train.shape[0], y_train.shape[1], 1)], 
            y=y_train.reshape(y_train.shape[0], y_train.shape[1], 1), 
            batch_size=config['batch_size'], 
            epochs=config['epochs'])

# Plot training history
trainer.plot_history()

# Run inference on test data
predictions = trainer.inference(X_test, indo_tokenizer, y_test)

# Evaluate the model
trainer.evaluate_translation(y_test, predictions)

