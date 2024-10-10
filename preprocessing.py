import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

def load_data():
    train_data = pd.read_json("DATA/MT_JAVNRF_INZNTV/train_preprocess.json")
    test_data = pd.read_json("DATA/MT_JAVNRF_INZNTV/test_preprocess.json")
    val_data = pd.read_json("DATA/MT_JAVNRF_INZNTV/valid_preprocess.json")

    for dataset in [train_data, test_data, val_data]:
        dataset.drop(columns="id", inplace=True)

    return train_data, test_data, val_data

def load_stopwords():
    indonesian_stopwords = stopwords.words("indonesian")
    javanese_stopwords = pd.read_csv("DATA/local_languages_stopwords.csv")
    javanese_stopwords.drop(columns=["indonesian", "sundanese"], inplace=True)

    return javanese_stopwords, indonesian_stopwords

def preprocess_text(text, stopwords):
    text = text.lower()
    text = text.strip('"')
    text = re.sub(r"[^\w\s]", "", text)
    text = "".join([char for char in text if char not in punctuation])
    text = " ".join(text.split())
    
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stopwords]
    
    preprocessed_text = " ".join(filtered_tokens)
    return preprocessed_text

def preprocess_data(data, javanese_stopwords, indonesian_stopwords):
    data["label"] = data["label"].apply(lambda x: preprocess_text(x, indonesian_stopwords))
    data["text"] = data["text"].apply(lambda x: preprocess_text(x, javanese_stopwords))
    data["text"] = data["text"].apply(lambda x: "START_ " + x + " _END")
    return data

def get_vocab(data):
    all_ind_words = set()
    all_javanese_words = set()
    
    for ind in data["label"]:
        all_ind_words.update(ind.split())
    
    for jav in data["text"]:
        all_javanese_words.update(jav.split())

    num_encoder_tokens = len(all_ind_words)
    num_decoder_tokens = len(all_javanese_words)

    return all_ind_words, all_javanese_words, num_encoder_tokens, num_decoder_tokens

def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def encode_sequences(tokenizer, length, lines):
    seq = tokenizer.texts_to_sequences(lines)
    seq = pad_sequences(seq, maxlen=length, padding="post")
    return seq

def main():
    # Load data
    train_data, test_data, val_data = load_data()

    # Load stopwords
    javanese_stopwords, indonesian_stopwords = load_stopwords()

    # Preprocess data
    train_data = preprocess_data(train_data, javanese_stopwords, indonesian_stopwords)
    test_data = preprocess_data(test_data, javanese_stopwords, indonesian_stopwords)
    val_data = preprocess_data(val_data, javanese_stopwords, indonesian_stopwords)

    # Combine data for vocabulary creation
    combined_data = pd.concat([train_data, test_data, val_data], ignore_index=True)

    # Get vocabulary
    all_ind_words, all_javanese_words, num_encoder_tokens, num_decoder_tokens = get_vocab(combined_data)

    # Tokenization
    java_tokenizer = tokenization(combined_data.text)
    indo_tokenizer = tokenization(combined_data.label)

    java_vocab_size = len(java_tokenizer.word_index) + 1
    indo_vocab_size = len(indo_tokenizer.word_index) + 1

    maxlength = combined_data["text"].apply(lambda x: len(x.split())).max()

    # Encode sequences
    X_train = encode_sequences(java_tokenizer, maxlength, train_data.text)
    y_train = encode_sequences(indo_tokenizer, maxlength, train_data.label)

    X_test = encode_sequences(java_tokenizer, maxlength, test_data.text)
    y_test = encode_sequences(indo_tokenizer, maxlength, test_data.label)

    X_val = encode_sequences(java_tokenizer, maxlength, val_data.text)
    y_val = encode_sequences(indo_tokenizer, maxlength, val_data.label)

    # Save processed data
    with open("processed_data.pickle", "wb") as f:
        pickle.dump(
            {
                "X_train": X_train,
                "X_test": X_test,
                "X_val": X_val,
                "y_train": y_train,
                "y_test": y_test,
                "y_val": y_val,
                "java_tokenizer": java_tokenizer,
                "indo_tokenizer": indo_tokenizer,
                "java_vocab_size": java_vocab_size,
                "indo_vocab_size": indo_vocab_size,
                "maxlength": maxlength,
                "num_encoder_tokens": num_encoder_tokens,
                "num_decoder_tokens": num_decoder_tokens,
            },
            f
        )

if __name__ == "__main__":
    main()
