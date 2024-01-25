import numpy as np
from sklearn.preprocessing import Normalizer
from bagofwordsModel import BoWModel

class TF_IDFModel:
    def fit_transform(self, sentences):
        bow = BoWModel()
        tf = bow.fit_transfrom(sentences)
        self.vocabulary = bow.vocabulary
        n_sentences = len(sentences)
        n_words = len(self.vocabulary)
        idf = np.array([[0. for _ in range(n_words)] for _ in range(n_sentences)])

        for i in range(n_sentences):
            for j in range(n_words):
                idf[i][j] = 1 + np.log((n_sentences + 1) / (np.count_nonzero(tf[:, j]) + 1))
        
        tfidf = Normalizer().fit_transform(np.multiply(idf, tf))

        return tfidf
