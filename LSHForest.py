from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_distances
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=100, centers=3, random_state=42, n_features=100)
print(X) # The dimensional values
print(y)  # What cluster the data points are from
# Create KNeighborsClassifier object with custom distance metric
knn = KNeighborsClassifier(n_neighbors=3, metric=cosine_distances)

# Fit the model
knn.fit(X, y)

# Query the model with a new data point
query_point = [[0, 0]]
nearest_neighbors = knn.kneighbors(query_point)
print(nearest_neighbors)
