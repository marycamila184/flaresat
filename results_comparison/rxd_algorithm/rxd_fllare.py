import matplotlib.pyplot as plt
import numpy as np

# Ordenar a lista de imagem e mascara pelo nome da imagem grande
# Filtrar os patches de uma imagem
# Ler a imagem grande
# Fazer os calculos
# For por patch
# Achar o patch da imagem menor
# Extrair o resultado em um vetor com o limiar
# Concatenar na lista
# Flatten 
# Resultado

THRESHOLD = 0.05 # Reference https://www.mdpi.com/2071-1050/15/6/5333 - 4.2. Anomaly Detection

image = np.random.rand(7000, 7000, [5,6])  # Example random data

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

anomaly_mask = distances_image > THRESHOLD
