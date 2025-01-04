import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class clustering:

    def fit(self, X, iterations, K):
        self.X = X
        self.iterations = iterations
        self.K = K
        self.m, self.n = X.shape
        self.initialize_centroids()
        self.cost_history = []
        for i in range(1, self.iterations+1):
            idx = self.find_closest_centroid()
            self.update_centroid(idx)
            self.cost_history.append(self.compute_cost(idx))
            print(f'Iteration: {i} | cost: {self.cost_history[i-1]}')
        self.cost_history = np.array(self.cost_history)
        return self.cost_history

    def initialize_centroids(self):
        self.new_centroids = np.zeros((self.K, self.n))
        for k in range(self.K):
            self.new_centroids[k] = self.X[np.random.choice(self.m)]

    def find_closest_centroid(self):
        distances = np.zeros((self.m, self.K))
        for k in range(self.K):
            distances[np.arange(self.m), k] = np.sqrt(np.sum((self.X - self.new_centroids[k])**2, axis=1))
        idx = np.argmin(distances, axis=1)
        return idx

    def update_centroid(self, ind):
        self.previous_centroids = self.new_centroids

        self.new_centroids = np.zeros((self.K, self.n))
        for k in range(self.K):
            centroid_points = self.X[ind == k]
            if centroid_points.shape[0] == 0:
                self.new_centroids[k] = self.X[np.random.choice(self.m)]
            else:
                self.new_centroids[k] = np.mean(centroid_points, axis=0)

    def compute_cost(self, ind):
        centroids = self.new_centroids[ind]
        distances = np.sqrt(np.sum((self.X - centroids)**2, axis=1))
        cost = np.mean(distances)
        return cost