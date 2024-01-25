import numpy as np
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r"[a-zA-Z0-9]+")

class BoWModel:
    def fit_transfrom(self, sentences):
        list_of_words = [] # should be set()
        words_in_sentences = []
        self.vocabulary = dict()

        for sentence in sentences:
            words = tokenizer.tokenize(sentence)
            words_cleaned = [w.lower() for w in words if len(w) > 1]
            words_in_sentences.append(words_cleaned)

            for word in words_cleaned:
                if word not in list_of_words:
                    list_of_words.append(word)
                    self.vocabulary[word] = 0

        list_of_words = sorted(list_of_words)
        n_words = len(list_of_words)
        n_sentences = len(sentences)

        for i in range(n_words):
            self.vocabulary[list_of_words[i]] = i

        matrix = [[0 for _ in range(n_words)] for _ in range(n_sentences)]
        for i in range(n_sentences):
            words = words_in_sentences[i]
            for word in words:
                matrix[i][self.vocabulary[word]] += 1

        return np.array(matrix)
