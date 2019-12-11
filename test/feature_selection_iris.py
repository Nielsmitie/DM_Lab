from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from skfeature.function.similarity_based.SPEC import spec
from skfeature.function.similarity_based import lap_score
from skfeature.function.sparse_learning_based.MCFS import mcfs
from skfeature.function.sparse_learning_based.NDFS import ndfs

iris = load_iris()

X, y = iris.data, iris.target

kbest = SelectKBest(chi2, k=3)
#lap = lap_score.lap_score(X)

X_new = kbest.fit_transform(X, y)

print("KBest: ", kbest.scores_)
#print("Laplacian: ", lap)
print("MCFS", mcfs(X, 3))
#print("NDFS", ndfs(X))
print("SPEC: ", spec(X))
