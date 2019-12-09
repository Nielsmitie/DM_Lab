from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

iris = load_iris()

X, y = iris.data, iris.target
kbest = SelectKBest(chi2, k=3)
X_new = kbest.fit_transform(X, y)

print(kbest.scores_)
