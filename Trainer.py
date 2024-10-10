from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model):
        self.model = model
        self.history = None

    def compile(self, optimizer=tf.keras.optimizers.AdamW(), loss='sparse_categorical_crossentropy', metrics=None):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print("Model compiled successfull")

    def fit(self, x, y, batch_size, epochs,X_val,y_val, validation_split=0.2, **kwargs):
        # ex: x = [X_train, y_train.reshape(y_train.shape[0], y_train.shape[1], 1)]
        #     y = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
        self.history = self.model.fit(
            x, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data = (X_val,y_val),
            **kwargs
        )

    def plot_history(self):
        if not self.history:
            print("Please fit the model first!")
        else:
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.legend(['train', 'validation'])
            plt.show()

    def save_model(self, path):
        self.model.save(path)


    def inference(self, x, tokenizer, sequences):
        # ex : x = [X_test, y_test.reshape(y_test.shape[0], y_test.shape[1], 1)]

        def get_word(n, tokenizer):
            for word, index in tokenizer.word_index.items():
                if index == n:
                    return word
            return None

        def decode_sequences(tokenizer, sequences):
            decoded_texts = []
            for sequence in sequences:
                decoded_words = []
                for int_val in sequence:
                    word = get_word(int_val, tokenizer)
                    if word is not None:
                        decoded_words.append(word)
                decoded_texts.append(' '.join(decoded_words))
            return decoded_texts

        y_preds = self.model.predict(x)
        preds_text = []
        for i, pred in enumerate(y_preds):
            temp = []
            for j, token_probs in enumerate(pred):
                predicted_word_index = np.argmax(token_probs)
                t = get_word(predicted_word_index, tokenizer)
                if j > 0:
                    if (t == get_word(np.argmax(pred[j - 1]), tokenizer)) or (t is None):
                        temp.append('')
                    else:
                        temp.append(t)
                else:
                    if t is None:
                        temp.append('')
                    else:
                        temp.append(t)

            preds_text.append(' '.join(temp))

            return decode_sequences(tokenizer, sequences)

    def evaluate_translation(self, y_true, y_pred):
        def preprocess_text(text):
            return set(word_tokenize(text.lower()))

        def calculate_exact_match_accuracy(y_true, y_pred):
            return accuracy_score(y_true, y_pred)

        def calculate_token_level_metrics(y_true, y_pred):
            true_tokens = [preprocess_text(text) for text in y_true]
            pred_tokens = [preprocess_text(text) for text in y_pred]

            tp, fp, fn = 0, 0, 0
            for true, pred in zip(true_tokens, pred_tokens):
                tp += len(true.intersection(pred))
                fp += len(pred - true)
                fn += len(true - pred)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            return precision, recall, f1

        def calculate_bleu_score(y_true, y_pred):
            bleu_scores = []
            for true, pred in zip(y_true, y_pred):
                reference = [word_tokenize(true.lower())]
                candidate = word_tokenize(pred.lower())
                bleu_scores.append(sentence_bleu(reference, candidate))
            return np.mean(bleu_scores)

        def evaluate_translation(y_true, y_pred):
            exact_match_accuracy = calculate_exact_match_accuracy(y_true, y_pred)
            precision, recall, f1 = calculate_token_level_metrics(y_true, y_pred)
            bleu_score = calculate_bleu_score(y_true, y_pred)

            return {
                'Exact Match Accuracy': exact_match_accuracy,
                'Token-level Precision': precision,
                'Token-level Recall': recall,
                'Token-level F1 Score': f1,
                'BLEU Score': bleu_score
            }

        results = evaluate_translation(y_true, y_pred)
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
