import os
import string

import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def get_data(file, title_text=True, subject=False, date=False):

    path = os.path.dirname(os.path.dirname(__file__)) + 'raw_data/'
    data = pd.read_csv(path + file)

    features = ["article"]

    data[["article"]] = data["title"] + ' ' + data["text"]

    if date:
        data[["date"]] = pd.to_datetime(data["date"])
        features.append("date")

    if subject:
        features.append("subject")

    X = data[features]
    y = data["target"]

    return X, y


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

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def lower(text):
    text = text.lower()
    return text

def number(text):
    text = ''.join(word for word in text if not word.isdigit())
    return text

def stop(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    text = [w for w in word_tokens if not w in stop_words]
    return text

def lemmatize(text):

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in text]
    text = lemmatized
    return text

def virg(text):
    text=" ".join(text)
    return text

if __name__ == "__main__":
    # data = get_titles('fake_real_data.csv')
    feat, target = get_data('fake_real_data.csv')
    print(feat.article[0])
    print(nl_preprocessing([feat.article[0]]))
