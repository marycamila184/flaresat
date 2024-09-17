import matplotlib.pyplot as plt
import numpy as np

# Example: Load your image data (shape: 7000 x 7000 x 10)
# Replace this with your actual image loading mechanism
image = np.random.rand(256, 256, 10)  # Example random data

# Reshape the image to (7000*7000, 10)
X = image.reshape(-1, 10)

# Compute the mean vector
mu = np.mean(X, axis=0)

# Compute the covariance matrix
cov_matrix = np.cov(X, rowvar=False)

# Inverse of the covariance matrix
cov_matrix_inv = np.linalg.inv(cov_matrix)

def mahalanobis_distance(x, mu, cov_matrix_inv):
    diff = x - mu
    return np.sqrt(np.dot(np.dot(diff, cov_matrix_inv), diff.T))

distances = np.zeros(X.shape[0])

for i in range(X.shape[0]):
    distances[i] = mahalanobis_distance(X[i], mu, cov_matrix_inv)

distances_image = distances.reshape(256, 256)

threshold = 3.85

anomaly_mask = distances_image > threshold

plt.imshow(anomaly_mask, cmap='gray')
plt.title('Anomaly Detection Mask')
plt.show()
