import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

expr = pd.read_csv("./SRP068591.tsv", sep="\t", index_col=0)
expr = np.log1p(expr)

common = expr.columns
X = expr.T  # samples x genes

kmeans = KMeans(n_clusters=4, random_state=0)
labels = kmeans.fit_predict(X)

out = pd.DataFrame(labels, index=common, columns=["cluster"])
out.to_csv("./results/kmeans_labels.csv")
print("Saved KMeans labels → ./results/kmeans_labels.csv")
