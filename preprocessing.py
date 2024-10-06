import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from sklearn.model_selection import train_test_split
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

    train_data.drop(columns="id", inplace=True)
    test_data.drop(columns="id", inplace=True)
    val_data.drop(columns="id", inplace=True)

    return train_data, test_data


def load_stopwords():
    indonesian_stopwords = stopwords.words("indonesian")
    javanese_stopwords = pd.read_csv("DATA\local_languages_stopwords.csv")
    javanese_stopwords.drop(columns=["indonesian", "sundanese"], inplace=True)

    return javanese_stopwords, indonesian_stopwords


def preprocess_text(text, stopwords):
    # lowercase
    text = text.lower()
    # remove quotes
    text = text.strip('"')
    # Combine removing special characters, punctuation, and extra whitespace
    text = re.sub(r"[^\w\s]", "", text)  # Removes special characters
    text = "".join(
        [char for char in text if char not in punctuation]
    )  # Removes punctuation
    text = " ".join(text.split())  # Removes extra whitespace

    # Tokenize
    tokens = word_tokenize(text)

    filtered_tokens = [word for word in tokens if word not in stopwords]

    # Join the tokens back into a string
    preprocessed_text = " ".join(filtered_tokens)

    return preprocessed_text


def preprocess_data(data, javanese_stopwords, indonesian_stopwords):
    data["label"] = data["label"].apply(
        lambda x: preprocess_text(x, indonesian_stopwords)
    )
    data["text"] = data["text"].apply(lambda x: preprocess_text(x, javanese_stopwords))
    data["text"] = data["text"].apply(lambda x: "START_ " + x + " _END")
    return data


def get_vocab(data):
    all_ind_words = set()
    for ind in data["label"]:
        for word in ind.split():
            if word not in all_ind_words:
                all_ind_words.add(word)

    all_javanese_words = set()
    for jav in data["text"]:
        for word in jav.split():
            if word not in all_javanese_words:
                all_javanese_words.add(word)

    return all_ind_words, all_javanese_words


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
    train_data, test_data = load_data()

    # load stopwords
    javanese_stopwords, indonesian_stopwords = load_stopwords()

    # preprocess data
    train_data = preprocess_data(train_data, javanese_stopwords, indonesian_stopwords)
    test_data = preprocess_data(test_data, javanese_stopwords, indonesian_stopwords)

    # Combine data (train and test data)
    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    combined_data = shuffle(combined_data)

    # Get vocabulary
    all_ind_words, all_javanese_words = get_vocab(combined_data)

    # Tokenization
    java_tokenizer = tokenization(combined_data.text)
    indo_tokenizer = tokenization(combined_data.label)

    maxlength = combined_data["text"].apply(lambda x: len(x.split())).max()

    # Encode sequences
    X = encode_sequences(java_tokenizer, maxlength, combined_data.text)
    y = encode_sequences(indo_tokenizer, maxlength, combined_data.label)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save processed data (Optional)
    with open("processed_data.pickle", "wb") as f:
        pickle.dump(
            {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "java_tokenizer": java_tokenizer,
                "indo_tokenizer": indo_tokenizer,
                "maxlength": maxlength,
            },
            f,
        )


if __name__ == "__main__":
    main()
