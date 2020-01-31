from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer


def get_dataset():
    """Load the 20 newsgroups dataset.
    
    Returns:
        (X, y, n) -- (dataset, labels, #classes)
    """    
    newsgroups = datasets.fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
    vectorizer = CountVectorizer(strip_accents="unicode", stop_words="english", max_features=2000)
    vectors = vectorizer.fit_transform(newsgroups.data)
    return vectors, newsgroups.target, 20
