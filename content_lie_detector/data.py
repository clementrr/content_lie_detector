import os
import string

import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def get_titles(file):
    path = os.path.dirname(os.path.dirname(__file__)) + 'raw_data/'
    data = pd.read_csv(path + file)
    data = data[['title', 'target']]

    return data


def nl_preprocessing(data, rm_punctuation=False, mk_lowercase=False, rm_digits=False, rm_stopwords=False, lemmatize=False):
    """Expects a pd series or 1D np array, of strings.
    Returns list of list of words."""
    output = []
    for txt in data:
        res = txt

        if rm_punctuation:
            res = ''.join(c for c in res if c not in string.punctuation)

        if mk_lowercase:
            res = res.lower()

        if rm_digits:
            res = ''.join(c for c in res if not c.isdigit())

        res = word_tokenize(res)
        # words = []
        # for s in sentences:
        # words.append(s.split(' '))
        # res = words

        if rm_stopwords:
            res = [w for w in res if w not in set(stopwords.words('english'))]

        if lemmatize:
            res = [WordNetLemmatizer().lemmatize(w) for w in res]

        output.append(res)

    return output


if __name__ == "__main__":
    # data = get_titles('fake_real_data.csv')
    data = get_titles('fake_real_data.csv')
    print(data.title[0])
    print(nl_preprocessing(data.title[0:2]))
