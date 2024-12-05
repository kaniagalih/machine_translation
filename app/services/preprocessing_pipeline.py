# 3 - app/services/preprocessing_pipeline.py

import pickle
import re
import csv
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set,Optional

from preprocessing import build_tokenizer
from preprocessing import preprocess_text
from preprocessing import add_start_end_tokens
from preprocessing import get_vocab
from preprocessing import encode_and_pad
from sklearn.model_selection import train_test_split, KFold


# Main processing pipeline
def process_pipeline(n_splits, random_state,vocab_save_dir="vocabularies"):
    Path(vocab_save_dir).mkdir(parents=True, exist_ok=True)

    combined_data_final = combined_data_new.copy()
    combined_data_final['label'] = combined_data_final.apply(lambda x: preprocess_text(x['label'], 'indo'), axis=1)
    combined_data_final['text'] = combined_data_final.apply(lambda x: preprocess_text(x['text'], 'java'), axis=1)
    # combined_data_final.to_csv("/content/drive/MyDrive/Dataset/csv-aksarajawa/dataset cerpen/combined_data_final_3.csv")
    add_start_end_tokens(combined_data_final,'label')
    combined_data_final['length_ind_sentence'] = combined_data_final['label'].apply(lambda x: len(x.split()))
    combined_data_final['length_jav_sentence'] = combined_data_final['text'].apply(lambda x: len(x.split()))

    combined_data_final = combined_data_final[
        (combined_data_final['length_ind_sentence'] <= 35) &
        (combined_data_final['length_ind_sentence'] > 2) &
        (combined_data_final['length_jav_sentence'] <= 35) &
        (combined_data_final['length_jav_sentence'] > 2)
    ]
    ind_vocab = get_vocab(combined_data_final, 'label')
    jav_vocab = get_vocab(combined_data_final, 'text')
    # Build tokenizers
    java_tokenizer = build_tokenizer(combined_data_final['text'])
    indo_tokenizer = build_tokenizer(combined_data_final['label'])

    print(f"Saving vocabularies to {vocab_save_dir}...")
    with open(f"{vocab_save_dir}/indonesian_vocab.pkl", 'wb') as f:
        pickle.dump(ind_vocab, f)
    with open(f"{vocab_save_dir}/javanese_vocab.pkl", 'wb') as f:
        pickle.dump(jav_vocab, f)

    print("Vocabularies saved successfully")

    input_token_index = java_tokenizer.word_index  # Javanese token index
    target_token_index = indo_tokenizer.word_index  # Indonesian token index

    max_length = max(combined_data_final['length_jav_sentence'].max(), combined_data_final['length_ind_sentence'].max())

    # Encode sequences
    X = encode_and_pad(java_tokenizer, combined_data_final['text'], max_length)
    y = encode_and_pad(indo_tokenizer, combined_data_final['label'], max_length)

    X_trainval,X_test,y_trainval,y_test = train_test_split(X,y,test_size = 0.1, random_state = 42)

    test_data = {
    'X_test': X_test,
    'y_test': y_test
    }

    # Save both X_test and y_test in a single pickle file
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)

    print("X_test and y_test have been saved in test_data.pkl")

    num_encoder_tokens = len(java_tokenizer.word_index) + 1  # Include OOV token
    java_length = max_length
    print('Java Vocabulary Size: %d' % num_encoder_tokens)

    # Prepare Indonesian tokenizer
    num_decoder_tokens = len(indo_tokenizer.word_index) + 1  # Include OOV token
    indo_length = max_length
    print('Indonesian Vocabulary Size: %d' % num_decoder_tokens)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_splits = []

    # Perform k-fold splitting
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_trainval)):
        print(f'\nFold {fold + 1}:')
        print(f'Training samples: {len(train_idx)}, Testing samples: {len(val_idx)}')

        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        fold_splits.append((X_train, X_val, y_train, y_val))
    X_train,X_val,y_train,y_val = fold_splits[3]
    # Create info dictionary
    tokenizer_info = {
        'max_length': max_length,
        'num_encoder_tokens': num_encoder_tokens,
        'num_decoder_tokens': num_decoder_tokens,
        'target_token_index': indo_tokenizer.word_index,
        'input_token_index': java_tokenizer.word_index,
        'java_tokenizer': java_tokenizer,
        'indo_tokenizer': indo_tokenizer
    }

    with open('dataset_kfolds.pkl','wb') as file:
      pickle.dump(fold_splits,file)
    with open('tokenizer_info.pkl','wb') as file:
      pickle.dump(tokenizer_info,file)

    print("Preprocessing Done!")

    return fold_splits, tokenizer_info,X_test,y_test


def is_potential_non_normalized(word: str) -> bool:
    """
    Check if a word potentially needs normalization

    Args:
        word (str): Word to check

    Returns:
        bool: True if word might need normalization
    """
    patterns = [
        r'(.)\1{2,}',  # Three or more repeated characters
        r'\d+',        # Contains numbers
        r'[A-Z]+',     # Contains uppercase (shouldn't exist after preprocessing)
        r'[!@#$%^&*(),.?":{}|<>]+',  # Contains special characters
        r'(.+?)\1{1,}',  # Repeated patterns
        r'2\b',        # Ends with '2' (common in Indonesian informal writing)
        r'[aiueo]{3,}'  # Three or more consecutive vowels
    ]

    return any(re.search(pattern, word) for pattern in patterns)

def find_non_normalized_words(vocab: Dict) -> Dict[str, List[str]]:
    """
    Find potentially non-normalized words in vocabulary

    Args:
        vocab (dict): Vocabulary dictionary

    Returns:
        dict: Dictionary of suspicious patterns and their matching words
    """
    suspicious_words = defaultdict(list)

    for word in vocab:
        # Check for repeated characters
        if re.search(r'(.)\1{2,}', word):
            suspicious_words['repeated_chars'].append(word)

        # Check for numbers
        if re.search(r'\d', word):
            suspicious_words['contains_numbers'].append(word)

        # Check for special characters
        if re.search(r'[^a-z\s-]', word):
            suspicious_words['special_chars'].append(word)

        # Check for very long words (possibly not properly split)
        if len(word) > 20:
            suspicious_words['very_long'].append(word)

        # Check for repeated patterns
        if re.search(r'(.+?)\1{1,}', word):
            suspicious_words['repeated_patterns'].append(word)

        # Check for multiple consecutive vowels
        if re.search(r'[aiueo]{3,}', word):
            suspicious_words['multiple_vowels'].append(word)

    return suspicious_words

def export_vocabularies(ind_vocab: Dict, jav_vocab: Dict, output_dir: str = "vocab_analysis") -> None:
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save raw vocabularies as pickle
    with open(f"{output_dir}/indonesian_vocab.pkl", 'wb') as f:
        pickle.dump(ind_vocab, f)
    with open(f"{output_dir}/javanese_vocab.pkl", 'wb') as f:
        pickle.dump(jav_vocab, f)

    # Find potentially non-normalized words
    ind_suspicious = find_non_normalized_words(ind_vocab)
    jav_suspicious = find_non_normalized_words(jav_vocab)

    # Create detailed CSV with all words and their analysis
    with open(f"{output_dir}/vocabulary_analysis.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Language', 'Word', 'Length', 'Suspicious_Patterns', 'Needs_Review', 'Source_Sentences'])

        # Process Indonesian vocabulary
        for word, sentences in ind_vocab.items():
            patterns = []
            if re.search(r'(.)\1{2,}', word): patterns.append('repeated_chars')
            if re.search(r'\d', word): patterns.append('contains_numbers')
            if re.search(r'[^a-z\s-]', word): patterns.append('special_chars')
            if len(word) > 20: patterns.append('very_long')
            if re.search(r'(.+?)\1{1,}', word): patterns.append('repeated_patterns')
            if re.search(r'[aiueo]{3,}', word): patterns.append('multiple_vowels')

            needs_review = len(patterns) > 0
            writer.writerow(['Indonesian', word, len(word),
                             '|'.join(patterns) if patterns else 'none',
                             'YES' if needs_review else 'NO',
                             '; '.join(sentences)])  # Join sentences for the word

        # Process Javanese vocabulary
        for word, sentences in jav_vocab.items():
            patterns = []
            if re.search(r'(.)\1{2,}', word): patterns.append('repeated_chars')
            if re.search(r'\d', word): patterns.append('contains_numbers')
            if re.search(r'[^a-z\s-]', word): patterns.append('special_chars')
            if len(word) > 20: patterns.append('very_long')
            if re.search(r'(.+?)\1{1,}', word): patterns.append('repeated_patterns')
            if re.search(r'[aiueo]{3,}', word): patterns.append('multiple_vowels')

            needs_review = len(patterns) > 0
            writer.writerow(['Javanese', word, len(word),
                             '|'.join(patterns) if patterns else 'none',
                             'YES' if needs_review else 'NO',
                             '; '.join(sentences)])  # Join sentences for the word

    # Create summary report
    with open(f"{output_dir}/normalization_summary.txt", 'w', encoding='utf-8') as f:
        f.write("Vocabulary Analysis Summary\n")
        f.write("=========================\n\n")

        f.write("Indonesian Vocabulary:\n")
        f.write(f"Total words: {len(ind_vocab)}\n")
        f.write("Potentially non-normalized words:\n")
        for pattern, words in ind_suspicious.items():
            f.write(f"\n{pattern}: {len(words)} words\n")
            f.write("Sample words: " + ", ".join(words[:10]) + "\n")

        f.write("\nJavanese Vocabulary:\n")
        f.write(f"Total words: {len(jav_vocab)}\n")
        f.write("Potentially non-normalized words:\n")
        for pattern, words in jav_suspicious.items():
            f.write(f"\n{pattern}: {len(words)} words\n")
            f.write("Sample words: " + ", ".join(words[:10]) + "\n")

def analyze_vocabularies(vocab_dir: str) -> None:
    """
    Load and analyze saved vocabularies

    Args:
        vocab_dir (str): Directory containing vocabulary pickle files
    """
    try:
        # Load vocabularies
        with open(f"{vocab_dir}/indonesian_vocab.pkl", 'rb') as f:
            ind_vocab = pickle.load(f)
        with open(f"{vocab_dir}/javanese_vocab.pkl", 'rb') as f:
            jav_vocab = pickle.load(f)

        # Export and analyze
        export_vocabularies(ind_vocab, jav_vocab, f"{vocab_dir}/analysis")

        print(f"\nAnalysis completed. Check the '{vocab_dir}/analysis' directory for:")
        print("- vocabulary_analysis.csv: Detailed analysis of each word")
        print("- normalization_summary.txt: Summary of potentially non-normalized words")
        print("- Raw vocabularies in pickle format")

    except FileNotFoundError:
        print(f"No vocabulary files found in {vocab_dir}")
    except Exception as e:
        print(f"Error analyzing vocabulary: {str(e)}")
        
def analyze_text_distribution(dataframe, text_column):
    # Calculate text lengths
    dataframe['text_length'] = dataframe[text_column].str.len()

    # Plot text length distribution
    plt.figure(figsize=(12, 4))

    # Subplot 1: Histogram of text lengths
    plt.subplot(131)
    dataframe['text_length'].hist(bins=50)
    plt.title('Text Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')

    # Subplot 2: Box plot of text lengths
    plt.subplot(132)
    sns.boxplot(x=dataframe['text_length'])
    plt.title('Text Length Boxplot')

    # Subplot 3: Descriptive statistics
    plt.subplot(133)
    length_stats = dataframe['text_length'].describe()
    plt.text(0.5, 0.5, str(length_stats),
             horizontalalignment='center',
             verticalalignment='center')
    plt.title('Length Statistics')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def analyze_vocabulary(dataframe, text_column):
    # Tokenize and count unique words
    all_words = ' '.join(dataframe[text_column]).split()
    word_freq = pd.Series(all_words).value_counts()

    # Plot word frequency
    plt.figure(figsize=(12, 4))

    # Top 50 most frequent words
    plt.subplot(121)
    word_freq[:50].plot(kind='bar')
    plt.title('Top 50 Most Frequent Words')
    plt.xticks(rotation=90)

    # Word frequency distribution
    plt.subplot(122)
    word_freq.plot(kind='hist', bins=50, log=True)
    plt.title('Word Frequency Distribution (Log Scale)')
    plt.xlabel('Frequency')
    plt.ylabel('Number of Words')

    plt.tight_layout()
    plt.show()

    return {
        'total_unique_words': len(word_freq),
        'top_10_words': word_freq[:10]
    }


# Get the data 
fold_splits, tokenizer_info,X_test,y_test = process_pipeline(n_splits = 5, random_state = 42)
# X_train, X_test, y_train, y_test, max_length,num_encoder_tokens,num_decoder_tokens,target_token_index,input_token_index,java_tokenizer,indo_tokenizer = process_pipeline()

analyze_vocabularies("../app/vocabularies")

with open('tokenizer_info.pkl','rb') as file:
  tokenizer_info = pickle.load(file)

with open('dataset_kfolds.pkl','rb') as file:
  fold_splits = pickle.load(file)

fold_splits[3]