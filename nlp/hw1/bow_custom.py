import numpy as np
from collections import Counter


# Helper function for creating bow dictionary
def create_bow_dictionary(vocab_text, bow_size):
    total_vocabulary = Counter(' '.join(vocab_text).split())
    k_common_words = total_vocabulary.most_common(bow_size)
    flattened_word_count = list(sum(k_common_words,()))
    bow_vocabulary = flattened_word_count[0:][::2]
    return bow_vocabulary


class Bow:
    def __init__(self, vocab_text, bow_size):
        self.bow_vocabulary = create_bow_dictionary(vocab_text, bow_size)

    def text_to_bow(self, text):
        token_counts = np.zeros(len(self.bow_vocabulary))
        for word in text.split():
            if word in self.bow_vocabulary:
                token_counts[self.bow_vocabulary.index(word)] += 1
        return np.array(token_counts, 'float32')