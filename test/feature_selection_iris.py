from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from model.competitors import get_competitor

iris = load_iris()

X, y = iris.data, iris.target

kbest = SelectKBest(chi2, k=3)

X_new = kbest.fit_transform(X, y)

print("KBest: ", kbest.scores_)

kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
print("Laplacian: ", get_competitor("LAP", X))
print("MCFS", get_competitor("MCFS", X))
print("NDFS", get_competitor("NDFS", X))
print("SPEC: ", get_competitor("SPEC", X))

