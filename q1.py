
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from random import sample
import numpy as np

centers = [(5, 10), (10, 5)]
data, cluster_id = make_blobs(n_samples=50, centers=centers, cluster_std=2.0) 
zipped = zip(data,cluster_id)
plt.scatter(data[:, 0], data[:, 1], s=50, c="blue", marker="s")
data_sample = np.array(sample(data.tolist(),20))
plt.scatter(data_sample[:, 0], data_sample[:, 1],s=75, c="red",marker="o")
plt.show()
