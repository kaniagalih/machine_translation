# 2 - app/services/preprocess_data.py

def load_data(path):
    return pd.read_json(path).drop(columns='id')

def load_dualisme_java_dictionary():
    # Load slang dictionary from CSV
    try:
      dualisme_dict = pd.read_csv("/content/drive/MyDrive/Dataset/MT-JavaIndo/nusa/dict_dualisme.csv")
      return dict(zip(dualisme_dict['dualisme'], dualisme_dict['usedword']))
    except FileNotFoundError as e:
      print(f"Warning: Slang dictionary file not found. Proceeding without slang replacement. Error: {e}")
      return {}

# Function to load slang dictionary
def load_slang_dictionary():
    # Load slang dictionary from CSV
    try:
        slang_dict_1 = pd.read_csv(
            "/content/drive/MyDrive/Dataset/MT-JavaIndo/new_kamusalay.csv",
            encoding="latin1",
            names=["slang", "formal"]  # Set custom column names
        )
        slang_dict_2 = pd.read_csv(
            "/content/drive/MyDrive/Dataset/MT-JavaIndo/inforformal-formal-Indonesian-dictionary.tsv",
            sep='\t', header=0
        )
        slang_dict_2 = slang_dict_2.rename(columns={'informal': 'slang'})
        slang_dict_3 = pd.read_csv('/content/drive/MyDrive/Dataset/MT-JavaIndo/dataset cerpen/kamus_alay_versi2.csv')
        slang_dict_3 = slang_dict_3.rename(columns={'Word': 'slang', 'formal word': 'formal'})

        # Combine the slang dictionaries
        slang_dict = pd.concat([slang_dict_1, slang_dict_2, slang_dict_3], ignore_index=True)
        return dict(zip(slang_dict['slang'], slang_dict['formal']))
    except FileNotFoundError as e:
        print(f"Warning: Slang dictionary file not found. Proceeding without slang replacement. Error: {e}")
        return {}

# Preprocessing the text
def preprocess_text(text,label):
    if not isinstance(text, str):
        return ""  # Return empty string if input is not valid

    if not isinstance(text, str):
      text = str(text)

    # Load slang dictionary
    if label == 'indo':
      slang_dict = load_slang_dictionary()
    elif label == 'java':
      slang_dict = load_dualisme_java_dictionary()

    # Lowercase the text
    text = text.lower()

    # Remove punctuation, URLs, HTML tags, emails
    text = re.sub(r'[%.,?!":;()[]"-=@©Ã$¨*‰]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    # Remove quotes
    text = text.strip('"').strip('“').strip('”')

    # # Handle word repetition
    # text = handle_word_repetition(text)

    # Replace slang words using the slang dictionary
    words = text.split()
    words = [
        slang_dict.get(word, slang_dict.get(word, word)) for word in words if word
    ]

    # Join words back into text
    text = ' '.join(words)

    # Remove special characters, punctuation, and extra whitespace
    text = re.sub(r"[^\w\s-]", "", text)
    text = "".join([char for char in text if char not in punctuation or char == '-'])
    text = " ".join(text.split())

    # Remove single characters
    text = ' '.join([w for w in text.split() if len(w) > 1])

    # Normalize and remove diacritics
    normalized_text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in normalized_text if not unicodedata.combining(c)])

    return text

def tokenize_combined_data_1(text):
      tokens = word_tokenize(text)

      # Join the tokens back into a string
      preprocessed_text = " ".join(tokens)

      return preprocessed_text


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
def encode_and_pad(tokenizer, data, length):
    sequences = tokenizer.texts_to_sequences(data)
    return pad_sequences(sequences, maxlen=length, padding='post')

# Function to build tokenizer
# def build_tokenizer(data):
#     tokenizer = Tokenizer(oov_token='<OOV>', filters='')
#     tokenizer.fit_on_texts(data)
#     return tokenizer

# from tensorflow.keras.preprocessing.text import Tokenizer

def build_tokenizer(data, lang='jv', vocab_size=20000):
    # Load BPEmb Javanese model
    bpemb = BPEmb(lang=lang, vs=vocab_size)

    # Tokenize data using BPEmb
    tokenized_data = [' '.join(bpemb.encode(text)) for text in data]

    # Create Keras Tokenizer
    tokenizer = Tokenizer(oov_token='<OOV>', filters='')
    tokenizer.fit_on_texts(tokenized_data)

    return tokenizer

def preprocess_final_data(text):
    if not isinstance(text, str):
        return ""

    # Load slang dictionary
    try:
        slang_dict = load_slang_dictionary()
    except FileNotFoundError:
        print("Warning: Slang dictionary file not found. Proceeding without slang replacement.")
        slang_dict = {}

    slang_dict = {key: str(value) for key, value in slang_dict.items()}
    words = text.split()
    words = [slang_dict.get(word, "") if slang_dict.get(word) == "null" else slang_dict.get(word,word) for word in words if word]
    text = ' '.join(words)

    return text


# check
sample_text = "hati2 hati! www.example.com"
processed_text = preprocess_text(sample_text,'indo')
print(processed_text)