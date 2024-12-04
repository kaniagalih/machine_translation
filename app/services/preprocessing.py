import os
import string
import pandas as pd
import numpy as np
import nltk
from bpemb import BPEmb

import re
from nltk.tokenize import sent_tokenize
from string import punctuation
import pickle
import csv
from collections import defaultdict
from typing import Dict, List, Set,Optional
from pathlib import Path
import unicodedata
from nltk.tokenize import word_tokenize

from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


import pickle
import re
import csv
from pathlib import Path
from collections import defaultdict

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import pandas as pd
import re
import unicodedata

@dataclass
class PreprocessingConfig:
    # Text Cleaning Parameters
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_urls: bool = True
    remove_html_tags: bool = True
    remove_emails: bool = True
    remove_quotes: bool = True
    min_word_length: int = 2
    max_sentence_length: int = 35

    # Slang Replacement
    enable_slang_replacement: bool = True
    slang_dictionaries: List[str] = field(default_factory=lambda: [
        '../app/data/new_kamusalay.csv',
        '../app/data/inforformal-formal-Indonesian-dictionary.tsv',
        '../app/data/kamus_alay_versi2.csv'
    ])

    # Normalization
    unicode_normalization: bool = True
    remove_diacritics: bool = True

    # Additional preprocessing specifics
    punctuation_to_remove: str = r'[%.,?!":;()[]"-=@©Ã$¨*‰]'
    special_char_regex: str = r"[^\w\s-]"

    # Tokenization
    tokenization_method: str = "word_tokenize"
    oov_token: str = "<OOV>"
    padding_type: str = "post"

    # Vocabulary
    max_vocab_size: Optional[int] = None
    min_word_frequency: int = 1

    # Data Splitting
    test_size: float = 0.1
    random_state: int = 42
    k_fold_splits: int = 5
    k_fold_shuffle: bool = True

    # Dataset Flags
    data_nusa_writes: bool = False
    data_nusax: bool = False
    data_korpus_nusantara: bool = False
    data_final: bool = False


class ConfigurablePreprocessor:
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.slang_dictionaries = {}

    def load_slang_dictionary(self, label: str = 'indo') -> Dict[str, str]:
      try:
          # Load Indonesian slang dictionaries
          if label == 'indo':
              slang_dicts = []
              for dict_path in self.config.slang_dictionaries:
                  try:
                      if dict_path.endswith('.csv'):  # Assuming CSV for most cases
                          df = pd.read_csv(dict_path, encoding='latin1', names=["slang", "formal"])
                          slang_dicts.append(df)
                      elif dict_path.endswith('.tsv'):  # For tab-separated values
                          df = pd.read_csv(dict_path, sep='\t', header=0)
                          df = df.rename(columns={'informal': 'slang'})  # Adjust column name
                          slang_dicts.append(df)
                  except FileNotFoundError:
                      print(f"Warning: Dictionary {dict_path} not found.")

              # Concatenate all dictionaries
              if slang_dicts:
                  slang_dict = pd.concat(slang_dicts, ignore_index=True)
                  return dict(zip(slang_dict['slang'], slang_dict['formal']))
              else:
                  print("No slang dictionaries were loaded.")
                  return {}

          # Load Javanese slang dictionary
          elif label == 'java':
              dualisme_dict = pd.read_csv("../app/data/dict_dualisme.csv")
              return dict(zip(dualisme_dict['dualisme'], dualisme_dict['usedword']))

      except Exception as e:
          print(f"Error loading slang dictionary: {e}")
          return pd.DataFrame()


    def preprocess_text(self, text: str, label: str) -> str:
        if not isinstance(text, str):
            return ""
        text = str(text)

        if self.config.enable_slang_replacement:
            slang_dict = self.load_slang_dictionary(label)
        else:
            slang_dict = {}

        if self.config.lowercase:
            text = text.lower()
        if self.config.remove_punctuation:
            text = re.sub(self.config.punctuation_to_remove, '', text)
        if self.config.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        if self.config.remove_html_tags:
            text = re.sub(r'<.*?>', '', text)
        if self.config.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        if self.config.remove_quotes:
            text = text.strip('"').strip('"').strip('"')

        words = text.split()
        words = [slang_dict.get(word, "") if slang_dict.get(word) == "<nan>" else slang_dict.get(word,word) for word in words if word]
        text = ' '.join(words)

        text = re.sub(self.config.special_char_regex, "", text)
        text = " ".join(text.split())

        if self.config.min_word_length > 1:
            text = ' '.join([w for w in text.split() if len(w) >= self.config.min_word_length])

        if self.config.unicode_normalization:
            normalized_text = unicodedata.normalize('NFKD', text)
            text = ''.join([c for c in normalized_text if not unicodedata.combining(c)])

        return text

    def call_data(self, config: PreprocessingConfig) -> pd.DataFrame:
        config = config or self.config  # Use self.config if no config is passed
        datasets = []
        if config.data_nusa_writes:
            try:
                data_nusa_alinea =  pd.read_csv("../app/data/nusa-alinea-15k.csv")
                data_nusa_alinea = data_nusa_alinea.rename(columns={'indo': 'Indonesian','jawa':'Javanese'})
                data_nusa_alinea = data_nusa_alinea[['Javanese', 'Indonesian'] + [col for col in data_nusa_alinea.columns if col not in ['Javanese', 'Indonesian']]]
                data_nusa_alinea = data_nusa_alinea.rename(columns={'Indonesian': 'label', 'Javanese': 'text'})
                datasets.append(data_nusa_alinea)
            except FileNotFoundError:
                print("Warning: nusa-alinea-15k.csv not found.")
        if config.data_nusax:
            try:
                datasets.append(pd.read_csv('../app/data/nusax_data.csv'))
            except FileNotFoundError:
                print("Warning: nusax_data.csv not found.")
        if config.data_korpus_nusantara:
            try:
                datasets.append(pd.read_csv('../app/data/korpus_nusantara_preprocessed.csv'))
            except FileNotFoundError:
                print("Warning: korpus_nusantara_preprocessed.csv not found.")
        if config.data_final:
            try:
                datasets.append(pd.read_csv("../app/data/final_data.csv"))
            except FileNotFoundError:
                print("Warning: final_data.csv not found.")

        if datasets:
            print("datasets has been combined, going to save as csv with name of combined_data_final.csv")
            dataset_final = pd.concat(datasets, ignore_index=True)
            # dataset_final.drop( axis=1, inplace=True)
            dataset_final.to_csv('../app/data/combined_data_final.csv')
            # raise ValueError("datasets has been combined, going to save as csv with name of combined_data_final.csv")
            return dataset_final
        else:
            raise ValueError("No datasets were loaded. Please check the configuration.")

import pandas as pd
from typing import Dict

class SlangLoader:
    def __init__(self, config):
        self.config = config

    def load_slang_dictionary(self, label: str = 'indo') -> pd.DataFrame:
        try:
            # Load Indonesian slang dictionaries
            if label == 'indo':
                slang_dicts = []
                for dict_path in self.config['slang_dictionaries']:  # Mengakses dictionary langsung
                    try:
                        if dict_path.endswith('.csv'):  # Assuming CSV for most cases
                            df = pd.read_csv(dict_path, encoding='latin1', names=["slang", "formal"])
                            slang_dicts.append(df)
                        elif dict_path.endswith('.tsv'):  # For tab-separated values
                            df = pd.read_csv(dict_path, sep='\t', header=0)
                            df = df.rename(columns={'informal': 'slang'})  # Adjust column name
                            slang_dicts.append(df)
                    except FileNotFoundError:
                        print(f"Warning: Dictionary {dict_path} not found.")

                # Concatenate all dictionaries into one DataFrame
                if slang_dicts:
                    combined_df = pd.concat(slang_dicts, ignore_index=True)
                    # Print the first few rows to verify
                    print("Combined slang dictionary loaded:")
                    print(combined_df.head())  # Show first few rows
                    return combined_df
                else:
                    print("No slang dictionaries were loaded.")
                    return pd.DataFrame()  # Return an empty DataFrame if no dictionaries loaded
        except Exception as e:
            print(f"Error loading slang dictionary: {e}")
            return pd.DataFrame()  # Return empty DataFrame in case of error

# Contoh pengujian dengan konfigurasi
config = {
    "slang_dictionaries": [
        "/content/drive/MyDrive/Dataset/MT-JavaIndo/new_kamusalay.csv",
        "/content/drive/MyDrive/Dataset/MT-JavaIndo/inforformal-formal-Indonesian-dictionary.tsv",
        "/content/drive/MyDrive/Dataset/MT-JavaIndo/dataset cerpen/kamus_alay_versi2.csv"
    ]
}

# Menginisialisasi objek dan memanggil fungsi
slang_loader = SlangLoader(config)
df_slang = slang_loader.load_slang_dictionary('indo')

# Jika ingin melihat seluruh DataFrame, bisa print seluruhnya
print("Full loaded slang dictionary:")
print(df_slang)


# Function to add start and end tokens to labels
def add_start_end_tokens(data, column):
    data[column] = data[column].apply(lambda x: f'<start> {x} <end>')

# Function to calculate vocabulary size
def get_vocab(data, column):
    vocab = defaultdict(list)
    for index, sentence in enumerate(data[column]):
        for word in sentence.split():
            vocab[word].append(sentence)  # Append the sentence to the word's list
    return vocab

# Function to encode and pad sequences
def encode_and_pad(tokenizer, data, length, padding_type='post',lang = None):
    vocab_size=10000
    bpemb = BPEmb(lang=lang, vs=vocab_size)

    # Tokenize data using BPEmb
    bpemb_data = [' '.join(bpemb.encode(text)) for text in data]
    sequences = tokenizer.texts_to_sequences(bpemb_data)
    return pad_sequences(sequences, maxlen = length, padding=padding_type)

def build_tokenizer(data,tokenization_method, oov_token, max_vocab_size,lang = None):
    # Load BPEmb Javanese model

    vocab_size=10000
    bpemb = BPEmb(lang=lang, vs=vocab_size)

    # Tokenize data using BPEmb
    tokenized_data = [' '.join(bpemb.encode(text)) for text in data]

    # Create Keras Tokenizer
    tokenizer = Tokenizer(oov_token=oov_token, filters='')
    tokenizer.fit_on_texts(tokenized_data)

    return tokenizer,bpemb

# Function to build tokenizer
# def build_tokenizer(data, tokenization_method, oov_token, max_vocab_size):
#     tokenizer = Tokenizer(oov_token=oov_token, filters='')
#     tokenizer.fit_on_texts(data)
#     if max_vocab_size:
#         tokenizer.word_index = {word: idx for word, idx in tokenizer.word_index.items() if idx <= max_vocab_size}
#     return tokenizer

def process_pipeline_with_config(
    config: Optional[PreprocessingConfig] = None,
    vocab_save_dir="vocabularies",
    selected_fold: Optional[int] = None  # None means use all folds (k-fold mode)
    ):
    # Create preprocessor with optional config
    preprocessor = ConfigurablePreprocessor(config or PreprocessingConfig())

    # Ensure the directory for saving vocabularies exists
    Path(vocab_save_dir).mkdir(parents=True, exist_ok=True)

    # Load data dynamically based on the configuration
    combined_data_final = preprocessor.call_data(None)

    # Apply preprocessing with configuration
    combined_data_final['label'] = combined_data_final.apply(
        lambda x: preprocessor.preprocess_text(x['label'], 'indo'),
        axis=1
    )
    combined_data_final['text'] = combined_data_final.apply(
        lambda x: preprocessor.preprocess_text(x['text'], 'java'),
        axis=1
    )

    # Add start and end tokens
    add_start_end_tokens(combined_data_final, 'label')

    # Add length information
    combined_data_final['length_ind_sentence'] = combined_data_final['label'].apply(lambda x: len(x.split()))
    combined_data_final['length_jav_sentence'] = combined_data_final['text'].apply(lambda x: len(x.split()))

    # Filter data based on configuration parameters
    combined_data_final = combined_data_final[
        (combined_data_final['length_ind_sentence'] <= config.max_sentence_length) &
        (combined_data_final['length_ind_sentence'] > config.min_word_length) &
        (combined_data_final['length_jav_sentence'] <= config.max_sentence_length) &
        (combined_data_final['length_jav_sentence'] > config.min_word_length)
    ]

    # Build tokenizers
    java_tokenizer,bpemb_jv = build_tokenizer(
        combined_data_final['text'], config.tokenization_method, config.oov_token, config.max_vocab_size,lang = 'jv'
    )
    # print("Java Tokenizer",java_tokenizer.word_index,len(java_tokenizer.word_index))
    indo_tokenizer, bpemb_id = build_tokenizer(
        combined_data_final['label'], config.tokenization_method, config.oov_token, config.max_vocab_size,lang = 'id'
    )
    # print("indo Tokenizer",indo_tokenizer.word_index,len(indo_tokenizer.word_index))

    # Save vocabularies
    with open(f"{vocab_save_dir}/indonesian_vocab.pkl", 'wb') as f:
        pickle.dump(indo_tokenizer.word_index, f)
    with open(f"{vocab_save_dir}/javanese_vocab.pkl", 'wb') as f:
        pickle.dump(java_tokenizer.word_index, f)

    # Determine the maximum length for padding
    max_length = max(combined_data_final['length_jav_sentence'].max(), combined_data_final['length_ind_sentence'].max())

    # Encode and pad the data
    X = encode_and_pad(java_tokenizer, combined_data_final['text'], max_length, config.padding_type,lang = 'jv')
    y = encode_and_pad(indo_tokenizer, combined_data_final['label'], max_length, config.padding_type,lang = 'id')

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print(X[1])
    print(y[1])

    # Split into train and test sets
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    # Perform k-fold splitting
    kfold = KFold(n_splits=config.k_fold_splits, shuffle=config.k_fold_shuffle, random_state=config.random_state)
    fold_splits = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_trainval)):
        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
        fold_splits.append((X_train, X_val, y_train, y_val))

    # Choose mode: k-fold or specific fold
    if selected_fold is None:
        # Use all folds
        print("Training using all folds (k-fold mode).")
        all_folds_data = []
        all_test_data = []
        for fold_num, (X_train, X_val, y_train, y_val) in enumerate(fold_splits):
            all_folds_data.append({
                'fold': fold_num,
                'X_train': X_train,
                'X_val': X_val,
                'y_train': y_train,
                'y_val': y_val
            })
        # Save datasets for all folds
        with open('dataset_kfolds.pkl', 'wb') as file:
            pickle.dump(all_folds_data, file)


        all_test_data.append({
            'X_test': X_test,
            'y_test': y_test
        })


        with open('dataset_test.pkl', 'wb') as file:
            pickle.dump(all_test_data, file)

        # Save dataset and tokenizer information
        tokenizer_info = {
            'max_length': max_length,
            'num_encoder_tokens': len(java_tokenizer.word_index) + 1,
            'num_decoder_tokens': len(indo_tokenizer.word_index) + 1,
            'target_token_index': indo_tokenizer.word_index,
            'input_token_index': java_tokenizer.word_index,
            'java_tokenizer': java_tokenizer,
            'indo_tokenizer': indo_tokenizer,
            'bpemb_jv':bpemb_jv,
            'bpemb_id':bpemb_id
        }
        with open('tokenizer_info.pkl', 'wb') as file:
            pickle.dump(tokenizer_info, file)

         # Print vocab sizes and splits
        print(f"Javanese vocabulary size: {len(java_tokenizer.word_index)}")
        print(f"Indonesian vocabulary size: {len(indo_tokenizer.word_index)}")
        print(f"Train set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")

        return all_folds_data, tokenizer_info, X_test, y_test

    else:
        # Use a specific fold
        if selected_fold < 0 or selected_fold >= config.k_fold_splits:
            raise ValueError(f"Invalid selected_fold: {selected_fold}. Must be in range 0 to {config.k_fold_splits - 1}.")

        X_train, X_val, y_train, y_val = fold_splits[selected_fold]

        trainval_data = {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val
        }

        with open('dataset_trainval.pkl','wb') as file:
            pickle.dump(trainval_data,file)

        print(f"Training using fold {selected_fold}.")

        # Save dataset and tokenizer information
        tokenizer_info = {
            'max_length': max_length,
            'num_encoder_tokens': len(java_tokenizer.word_index) + 1,
            'num_decoder_tokens': len(indo_tokenizer.word_index) + 1,
            'target_token_index': indo_tokenizer.word_index,
            'input_token_index': java_tokenizer.word_index,
            'java_tokenizer': java_tokenizer,
            'indo_tokenizer': indo_tokenizer,
            'bpemb_jv':bpemb_jv,
            'bpemb_id':bpemb_id
        }
        with open('tokenizer_info.pkl', 'wb') as file:
            pickle.dump(tokenizer_info, file)

        print(f"Javanese vocabulary size: {len(java_tokenizer.word_index)}")
        print(f"Indonesian vocabulary size: {len(indo_tokenizer.word_index)}")
        print(f"Train set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")

        return tokenizer_info,X_train, X_val, y_train, y_val, X_test, y_test

import yaml
from dataclasses import asdict



def save_config_to_yaml(config: PreprocessingConfig, file_path: str):
    """
    Save the preprocessing configuration to a YAML file.

    Args:
        config (PreprocessingConfig): The configuration object.
        file_path (str): Path to the YAML file.
    """
    with open(file_path, "w") as yaml_file:
        yaml.dump(asdict(config), yaml_file, default_flow_style=False)
    print(f"Configuration saved to {file_path}")

def load_config_from_yaml(file_path: str) -> PreprocessingConfig:
    """
    Load a preprocessing configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        PreprocessingConfig: The loaded configuration object.
    """
    with open(file_path, "r") as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
    return PreprocessingConfig(**config_dict)

def main():
    # Set selected fold: choose specific fold or None for k-fold
    selected_fold = None

    # Define a custom configuration
    custom_config = PreprocessingConfig(
        #Preprocessing
        lowercase=True,
        remove_urls=True,
        remove_punctuation=True,
        remove_html_tags=True,
        remove_emails=True,
        remove_quotes=True,
        #Tokenizer tuning
        min_word_length=3,
        enable_slang_replacement=True,
        max_sentence_length=35,
        max_vocab_size=30000,
        min_word_frequency=2,
        #Split and Folds
        test_size=0.2,
        random_state=42,
        k_fold_splits=5,

        k_fold_shuffle=True,
        #Dataset
        data_nusa_writes = True,
        data_nusax = False,
        data_korpus_nusantara = False,
        data_final = False

    )

    # Save the configuration as YAML
    config_file = "custom_preprocessing_config.yaml"
    save_config_to_yaml(custom_config, config_file)

    # Load the configuration from YAML
    loaded_config = load_config_from_yaml(config_file)

    # Process the pipeline with the loaded configuration
    if selected_fold is None:
        all_folds_data, tokenizer_info, X_test, y_test = process_pipeline_with_config(config=loaded_config)
        # Save the pipeline results (optional)
        with open("pipeline_results.pkl", "wb") as f:
            pickle.dump({
                "fold_splits": all_folds_data,
                "tokenizer_info": tokenizer_info,
                "X_test": X_test,
                "y_test": y_test,
            }, f)
        print("Pipeline processed for all folds (k-fold mode) and results saved.")
    else:
        if selected_fold < 0 or selected_fold >= custom_config.k_fold_splits:
            raise ValueError(f"Invalid selected_fold: {selected_fold}. Must be in range 0 to {custom_config.k_fold_splits - 1}.")

        tokenizer_info, X_train, X_val, y_train, y_val, X_test, y_test = process_pipeline_with_config(
            config=loaded_config, selected_fold=selected_fold
        )
        # Save the pipeline results (optional)
        with open(f"pipeline_results-selected_fold_{selected_fold}.pkl", "wb") as f:
            pickle.dump({
                "tokenizer_info": tokenizer_info,
                "X_train": X_train,
                "X_val": X_val,
                "y_train": y_train,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
                'bpemb_jv':bpemb_jv,
                'bpemb_id':bpemb_id
            }, f)
        print(f"Pipeline processed for selected fold {selected_fold} and results saved.")

if __name__ == "__main__":
    main()
