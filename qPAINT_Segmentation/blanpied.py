import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

def get_cluster_randomized_2d(points):
    ## Function translated by Max Winga and ChatGPT from MATLAB code previously 
    ## written by Ai-Hui Tang, PhD (tangah@ustc.edu.cn), on 12/30/2018.
    ## Link to ChatGPT Conversation:
    ## https://chat.openai.com/share/943e6e3d-a069-4d58-bf42-7ea8ff5a52be

    # Check that points is long enough, else return empty
    if len(points) < 3:
        return np.empty((0, 2)) 
    
    # Calculate the convex hull
    hull = ConvexHull(points)
    hull_path = Path(points[hull.vertices])

    # Generate random points
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    num_points = len(points)
    random_points_inside = np.empty((0, 2))

    # Keep generating points until we have enough
    while len(random_points_inside) < num_points:
        remaining_points = num_points - len(random_points_inside)
        random_points = np.column_stack((
            np.random.uniform(min_x, max_x, remaining_points * 10),  # Overgenerate and filter
            np.random.uniform(min_y, max_y, remaining_points * 10)
        ))

        # Filter points inside the convex hull
        inside_hull = hull_path.contains_points(random_points)
        random_points_inside = np.vstack([random_points_inside, random_points[inside_hull]])

    # Adjust the number of points to match the input
    adjusted_random_points = random_points_inside[:num_points]

    return adjusted_random_points

def as_tuple(point):
    return (point[0], point[1])

def distance_squared(points, i1, i2):
    return (points[i1][0] - points[i2][0])**2 + (points[i1][1] - points[i2][1])**2

def find_max_index_above_threshold(arr, threshold):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] > threshold:
            low = mid + 1
        else:
            high = mid - 1
    return high

def blanpied_clustering(points, peak_distance_threshold, density_factor=2.5, 
                        eps_multiplier=5, min_samples=60, min_cluster_size=3):
    
    # Calculate full_MMD for whole frame
    kdtree = KDTree(points)
    distances, _ = kdtree.query(points, k=2, workers=-1)
    mmd = np.mean(distances[:, 1])
    
    # Use DBSCAN to locate "synaptic clusters" 60 min_samples, 5 * full_MMD eps
    db = DBSCAN(eps=eps_multiplier*mmd, min_samples=min_samples, metric='euclidean', n_jobs=-1).fit(points)
    synaptic_clusters = []
    for label in np.unique(db.labels_):
        if label == -1: continue
        synaptic_clusters.append(np.where(db.labels_ == label)[0])
        
    nanocluster_groups = []
    # For each synaptic cluster
    for synaptic_cluster in synaptic_clusters:
        
        # Calculate local_density_radius
        local_kdtree = KDTree(points[synaptic_cluster])
        local_mmd = np.mean(distances[synaptic_cluster, 1])
        local_density_radius = density_factor * local_mmd

        # Calculate Local Densities using points within local_density_radius
        local_densities = np.array(local_kdtree.query_ball_point(points[synaptic_cluster], local_density_radius, 
                                                                 workers=-1, return_length=True))
        
        # Sort the indices in synaptic_cluster by local_density
        sorted_indices = np.argsort(-local_densities)
        local_densities = local_densities[sorted_indices]
        synaptic_cluster = synaptic_cluster[sorted_indices]

        # Get Randomized Points and Calculate LD0_threshold
        LD0_thresholds = []
        for _ in range(5):
            r_points = get_cluster_randomized_2d(points[synaptic_cluster])
            r_kdtree = KDTree(r_points)
            r_local_densities = np.array(r_kdtree.query_ball_point(r_points, local_density_radius, 
                                                                   workers=-1, return_length=True))
            LD0_thresholds.append(np.mean(r_local_densities) + 4 * np.std(r_local_densities))
        LD0_threshold = np.median(np.array(LD0_thresholds))

        # Find nanoclusters
        max_index = find_max_index_above_threshold(local_densities, LD0_threshold) + 1
        points_over_threshold = points[synaptic_cluster[:max_index]]
        thresh_kdtree = KDTree(points_over_threshold)
        nanoclusters = {}
        nearby_points = np.array(thresh_kdtree.query_ball_point(points_over_threshold, 
                                                                peak_distance_threshold, 
                                                                workers=-1))
        for i in range(max_index):
            nearby_peaks = [point for point in nearby_points[i] if point in nanoclusters]
            if len(nearby_peaks) == 0:
                nanoclusters[i] = [i]
            else:
                peak_distances = np.array([distance_squared(points_over_threshold, peak, i) for peak in nearby_peaks])
                nearest_peak = nearby_peaks[np.argmin(peak_distances)]
                nanoclusters[nearest_peak].append(i)
        
        final_nanoclusters = {}
        # Filter out Subclusters
        for center in nanoclusters:
            cluster_indices = np.array(nanoclusters[center])
            cluster_kdtree = KDTree(points[synaptic_cluster[cluster_indices]])
            final_indices = np.array([False for i in range(len(cluster_indices))])
            final_indices[0] = True
            to_check = np.array([0])
            while len(to_check) > 0:
                within_range = cluster_kdtree.query_ball_point(points_over_threshold[cluster_indices[to_check]], 
                                                               r=2*local_mmd, workers=-1)
                to_check = []
                for point_list in within_range:
                    for point in point_list:
                        if not final_indices[point]:
                            final_indices[point] = True
                            to_check.append(point)
            cluster_indices = cluster_indices[final_indices]

            # Discard cluster if small, else update the cluster in final index coords
            if not len(cluster_indices) < min_cluster_size: 
                final_nanoclusters[synaptic_cluster[center]] = [synaptic_cluster[idx] for idx in cluster_indices]

        # Add to nanocluster_groups
        nanocluster_groups.append(final_nanoclusters)
    
    return synaptic_clusters, nanocluster_groups