from clusters import Cluster

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import HDBSCAN

class ClusteringAlgorithm:
    _instances = []
    _next_label = 0

    def __init__(self, clustering_function, target_points, label=None):
        """
        Initialize a ClusteringAlgorithm instance.

        Args:
            clustering_function (callable): The clustering function to be used.
            target_points: The target points for clustering.
            label (str, optional): Custom label for the instance. Defaults to None.
        """
        if not callable(clustering_function):
            raise TypeError("clustering_function must be callable")
        self.clustering_function = clustering_function
        self.target_points = target_points

        if label is None:
            self._label = str(ClusteringAlgorithm._next_label)
            ClusteringAlgorithm._next_label += 1
        else:
            if label.isdigit():
                raise ValueError("Custom label cannot be an integer")
            self._label = label

        ClusteringAlgorithm._instances.append(self)

    def __call__(self, *args, **kwargs):
        """
        Call the clustering function with the provided arguments.

        Returns:
            The result of the clustering function.
        """
        try:
            result = self.clustering_function(*args, **kwargs)
            return result
        except Exception as e:
            raise Exception(f"Error in ClusteringAlgorithm {self.label}: {str(e)}")

    def __eq__(self, other):
        """
        Check if two ClusteringAlgorithm instances are equal based on their labels.

        Args:
            other: Another ClusteringAlgorithm instance to compare with.

        Returns:
            True if the labels are equal, False otherwise.
        """
        if isinstance(other, ClusteringAlgorithm):
            return self.label == other.label
        return False

    def __hash__(self):
        """
        Return the hash value of the ClusteringAlgorithm instance based on its label.

        Returns:
            The hash value of the label.
        """
        return hash(self.label)

    @classmethod
    def get_instances(cls):
        """
        Get all instances of ClusteringAlgorithm.

        Returns:
            A list of all ClusteringAlgorithm instances.
        """
        return cls._instances

    @classmethod
    def get_instance_by_label(cls, label):
        """
        Get a ClusteringAlgorithm instance by its label.

        Args:
            label (str): The label of the instance to retrieve.

        Returns:
            The ClusteringAlgorithm instance with the specified label, or None if not found.
        """
        for instance in cls._instances:
            if instance.label == label:
                return instance
        return None

    @property
    def label(self):
        """
        Get the label of the ClusteringAlgorithm instance.

        Returns:
            The label of the instance.
        """
        return self._label

    @label.setter
    def label(self, value):
        """
        Set the label of the ClusteringAlgorithm instance.

        Args:
            value (str): The new label value.

        Raises:
            ValueError: If the provided label is an integer.
        """
        if value.isdigit():
            raise ValueError("Custom label cannot be an integer")
        self._label = value

    def __repr__(self):
        """
        Return a string representation of the ClusteringAlgorithm instance.

        Returns:
            A string representation of the instance.
        """
        return f"ClusteringAlgorithm(label={self.label})"

    def __del__(self):
        """
        Remove the ClusteringAlgorithm instance from the list of instances when deleted or garbage-collected.
        """
        ClusteringAlgorithm._instances.remove(self)

def filter_by_nth_neighbor_distance(points, n, max_distance):
    kdtree = KDTree(points.points)
    distances = np.array(kdtree.query(points.points, k=n, workers=-1)[0][:, -1])
    indices = np.where(distances < max_distance)
    return Cluster(points, indices, label=points.label), distances[indices]

def get_clusters_kmeans(points, k_max=10, to_plot=False, min_cluster_size=5):
    # Range of k values
    k_max = min(k_max, len(points) -1)
    k_values = np.array(range(1, k_max+1))
    sse = []

    # Calculating SSE for each k
    for k in k_values:
        centroids, distortion = kmeans(points, k, seed=0)
        sse.append(distortion)

    elbow = find_elbow(k_values, sse, to_plot=to_plot)
    if elbow == -1:
        return get_clusters_kmeans(points, k_max*2, to_plot)
    
    differences = []
    for i in range(len(sse) - 1):
        differences.append(sse[i+1] - sse[i])
    differences = np.array(differences)
    
    if to_plot:
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(k_values, sse, 'bx-')
        plt.xlabel('k')
        plt.ylabel('SSE')
        plt.axvline(elbow, linestyle='--', label=f"Optimal k = {elbow}")
        plt.title("Elbow Method For Optimal k")

        plt.subplot(1, 2, 2)
        plt.plot(k_values[:-1], differences, 'go-')
        plt.xlabel('k')
        plt.ylabel('Different Between Points')
        
        plt.tight_layout()
        plt.show()

    centroids = kmeans(points, elbow, seed=0)[0]
    cluster_indices = vq(points, centroids)[0]
    clusters = []
    for i in range(elbow):
        cluster = Cluster(points, np.where(cluster_indices == i), label=points.label)
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
    return centroids, clusters

def find_elbow(k_values, sse, to_plot=True):
    max_val = max(sse)
    scaled_sse = [err/max_val for err in sse]
    ratios = [0]
    for i in range(1, len(sse) - 1):
        before_avg = (scaled_sse[i] - scaled_sse[0])/i
        after_avg = (scaled_sse[len(scaled_sse)-1] - scaled_sse[i])/(len(scaled_sse)- i - 1)
        ratios.append(before_avg / after_avg)
    ratios.append(0)
    if to_plot:
        plt.figure()
        plt.plot(k_values, ratios, 'bx-')
        plt.show()
    return k_values[np.argmax(ratios)]

def kmeans_function(points, n, max_distance, k_max, min_total_points=10):
    kdtree = KDTree(points.points)
    distances = np.array(kdtree.query(points.points, k=n, workers=-1)[0][:, -1])

    if type(n) is not list:
        n = [n, n]
    if type(max_distance) is not list:
        max_distance = [max_distance, max_distance]
    if len(points) >= min_total_points:
        points, distances = filter_by_nth_neighbor_distance(points, n[0], max_distance[0])
    else:
        return []
    if len(points) >= min_total_points:
        points, distances = filter_by_nth_neighbor_distance(points, n[1], max_distance[1])
    else:
        return []
    if len(points) >= min_total_points:
        centroids, clusters = get_clusters_kmeans(points, k_max=k_max, to_plot=False)
        return clusters
    else:
        return []

def kmeans_function_generator(n, max_distance, k_max):
    def clustering_function(points):
        return kmeans_function(points, n, max_distance, k_max)
    return clustering_function

def nth_neighbor_hdbscan(points, n, max_distance, min_total_points=10, min_cluster_size=10, 
                         min_samples=None, cluster_selection_epsilon=0.0):
    if type(n) is not list:
        n = [n, n]
    if type(max_distance) is not list:
        max_distance = [max_distance, max_distance]
    if len(points) >= min_total_points:
        points, _ = filter_by_nth_neighbor_distance(points, n[0], max_distance[0])
    else:
        return []
    if len(points) >= min_total_points:
        points, _ = filter_by_nth_neighbor_distance(points, n[1], max_distance[1])
    else:
        return []
    if len(points) >= min_total_points:
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            cluster_selection_epsilon=cluster_selection_epsilon)
        labels = clusterer.fit_predict(points.points)

        # Create a list to store the clusters
        clusters = []

        # Iterate over the unique cluster labels
        for label in np.unique(labels):
            if label != -1:  # Exclude noise points (labeled as -1)
                # Get the indices of points belonging to the current cluster
                cluster_indices = np.where(labels == label)[0]
                
                # Create a new Cluster object for the current cluster
                cluster = Cluster(points, cluster_indices, label=points.label)
                
                # Add the cluster to the list
                clusters.append(cluster)

        return clusters
    else:
        return []

def nth_neighbor_hdbscan_function_generator(n, max_distance, min_total_points=10, min_cluster_size=10, 
                                            min_samples=None, cluster_selection_epsilon=0.0):
    def clustering_function(points):
        return nth_neighbor_hdbscan(points, n, max_distance, min_total_points, min_cluster_size,
                                    min_samples, cluster_selection_epsilon)
    return clustering_function

def hdbscan_clustering(points, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0):
    # Extract the coordinates from the points
    coords = points.points
    if len(coords) < 10:
        return []
    
    # Perform HDBSCAN clustering
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            cluster_selection_epsilon=cluster_selection_epsilon)
    labels = clusterer.fit_predict(points.points)
        
    # Create a list to store the clusters
    clusters = []

    # Create a list to store the clusters
    clusters = []

    # Iterate over the unique cluster labels
    for label in np.unique(labels):
        if label != -1:  # Exclude noise points (labeled as -1)
            # Get the indices of points belonging to the current cluster
            cluster_indices = np.where(labels == label)[0]
            
            # Create a new Cluster object for the current cluster
            cluster = Cluster(points, cluster_indices, label=points.label)
            
            # Add the cluster to the list
            clusters.append(cluster)

    return clusters

def hdbscan_function_generator(min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0):
    def clustering_function(points):
        return hdbscan_clustering(points, min_cluster_size, min_samples, cluster_selection_epsilon)
    return clustering_function