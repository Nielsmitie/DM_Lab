from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer


def get_dataset():
    newsgroups = datasets.fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
    vectorizer = CountVectorizer(strip_accents="unicode", stop_words="english", max_features=2000)
    vectors = vectorizer.fit_transform(newsgroups.data)
    return vectors, newsgroups.target, 20
