from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer


def get_dataset():
    newsgroups = datasets.fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(newsgroups.data)
    return vectors, newsgroups.target, 20
