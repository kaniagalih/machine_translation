import pickle
import yaml
import os

def load_processed_data(file_path='processed_data.pickle'):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def generate_config(processed_data, output_file='config/config.yaml'):
    config = {
        "maxlength": processed_data['maxlength'],
        "num_encoder_tokens": processed_data['num_encoder_tokens'],
        "num_decoder_tokens": processed_data['num_decoder_tokens'],
        "indo_vocab_size": processed_data['indo_vocab_size'],
        "java_vocab_size": processed_data['java_vocab_size'],
        
        # Model parameters 
        "latent_dim": 256,
        "attention": True,
        
        # Training parameters 
        "batch_size": 64,
        "epochs": 20,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"],
        
        # Data shapes (for reference)
        "X_train_shape": processed_data['X_train'].shape,
        "y_train_shape": processed_data['y_train'].shape,
        "X_test_shape": processed_data['X_test'].shape,
        "y_test_shape": processed_data['y_test'].shape,
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config

def main():
    if not os.path.exists('processed_data.pickle'):
        print("Processed data not found. Please run preprocessing.py first.")
        return
    
    processed_data = load_processed_data()
    config = generate_config(processed_data)
    
    print("Configuration file generated: config.yaml")
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()