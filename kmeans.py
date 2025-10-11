import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# Load your log-transformed expression matrix
expr = pd.read_csv("./SRP068591.tsv", sep="\t", index_col=0)
expr = np.log1p(expr)

# Keep only the samples used for clustering
common = expr.columns
X = expr.T  # samples x genes

# Run KMeans (for example, k=4)
kmeans = KMeans(n_clusters=4, random_state=0)
labels = kmeans.fit_predict(X)

# Save to CSV in the expected format
out = pd.DataFrame(labels, index=common, columns=["cluster"])
out.to_csv("./results/kmeans_labels.csv")
print("Saved KMeans labels → ./results/kmeans_labels.csv")
