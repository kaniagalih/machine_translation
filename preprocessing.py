import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from string import punctuation
import pandas as pd

nltk.download('stopwords', quiet=True)
indonesian_stopwords = stopwords.words('indonesian')

javanese_stopwords = pd.read_csv("local_languages_stopwords.csv")
javanese_stopwords.drop(columns=["indonesian", "sundanese"], inplace=True)


train_data = pd.read_json("DATA\MT_JAVNRF_INZNTV\train_preprocess.json")
test_data = pd.read_json("DATA\MT_JAVNRF_INZNTV\test_preprocess.json")
val_data = pd.read_json("DATA\MT_JAVNRF_INZNTV\valid_preprocess.json")

train_data.drop(columns='id', inplace=True)
test_data.drop(columns='id', inplace=True)
val_data.drop(columns='id', inplace=True)


def preprocess_text(text, language_label):
    #lowercase
    text = text.lower()
    #remove quotes
    text = text.strip('"')
    # Combine removing special characters, punctuation, and extra whitespace
    text = re.sub(r"[^\w\s]", "", text)  # Removes special characters
    text = "".join([char for char in text if char not in punctuation])  # Removes punctuation
    text = " ".join(text.split())  # Removes extra whitespace
    #Tokenize
    tokens = word_tokenize(text)

    # Select stopwords based on language label
    if language_label == "label":
        stopwords = indonesian_stopwords
    elif language_label == "text":
        stopwords = javanese_stopwords
    else:
        # Default to empty set if language label is not recognized
        stopwords = set()

    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stopwords]


    # Join the tokens back into a string
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text

def get_vocab(train_data):
    all_ind_words=set()
    for ind in train_data['label']:
        for word in ind.split():
            if word not in all_ind_words:
                all_ind_words.add(word)

    all_javanese_words=set()
    for jav in train_data['text']:
        for word in jav.split():
            if word not in all_javanese_words:
                all_javanese_words.add(word)

    input_words = sorted(list(all_ind_words))
    target_words = sorted(list(all_javanese_words))
    num_encoder_tokens = len(all_ind_words)
    num_decoder_tokens = len(all_javanese_words)

    return input_words,target_words,num_encoder_tokens,num_decoder_tokens

def prepare_dataset(train_data,test_data,val_data):
    train_data['label'] = train_data['label'].apply(lambda x: preprocess_text(x, indonesian_stopwords))
    train_data['text'] = train_data['text'].apply(lambda x: preprocess_text(x, javanese_stopwords))
    test_data['label'] = test_data['label'].apply(lambda x: preprocess_text(x, indonesian_stopwords))
    test_data['text'] = test_data['text'].apply(lambda x: preprocess_text(x, javanese_stopwords))
    val_data['label'] = val_data['label'].apply(lambda x: preprocess_text(x, indonesian_stopwords))
    val_data['text'] = val_data['text'].apply(lambda x: preprocess_text(x, javanese_stopwords))

    train_data['text'] = train_data['text'].apply(lambda x : 'START_ '+ x + ' _END')
    test_data['text'] = test_data['text'].apply(lambda x : 'START_ '+ x + ' _END')
    val_data['text'] = val_data['text'].apply(lambda x : 'START_ '+ x + ' _END')

    maxlength = max(train_data['length_jav_sentence'])















