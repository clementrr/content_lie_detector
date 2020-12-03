from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.sequence import pad_sequences


class CustomTokenizer(TransformerMixin, BaseEstimator):

    def __init__(self, min_occurence=15, max_len=600):
        self.min_occurence = min_occurence
        self.max_len = max_len
        self.word_if_dict = None
        self.vocab_size = None
        self.occurences_dictionnary = None
        return None

    def fit(self, X, y=None):
        X = [sentence.split(' ') for sentence in X]
        word_to_id = {}
        iter_ = 1
        for sentence in X:
            for word in sentence:
                if word in word_to_id:
                    continue
                word_to_id[word] = iter_
                iter_ += 1
        self.word_id_dict = word_to_id
        self.vocab_size = len(self.word_id_dict)

        occurences = {}
        for s in X:
            for w in s:
                if w not in occurences:
                    occurences[w] = 0
                occurences[w] += 1

        self.occurences_dictionnary = occurences
        return self

    def transform(self, X, y=None):
        X = [sentence.split(' ') for sentence in X]
        frequent_words = {k: v for k, v in self.occurences_dictionnary.items() if v >= self.min_occurence}
        X = [[word for word in sentence if word in frequent_words] for sentence in X]
        X = [[self.word_id_dict[_] for _ in s if _ in self.word_id_dict] for s in X]
        X = pad_sequences(X, dtype='float32', padding='post', maxlen=self.max_len)
        return X
