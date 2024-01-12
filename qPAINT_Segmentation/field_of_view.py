import numpy as np
import math as m
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import tifffile
import pandas as pd
from sklearn.cluster import DBSCAN
import warnings
import csv

from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from stardist.models import StarDist2D
from csbdeep.utils import normalize

from plot_helpers import plot_scale_bar, PlotColors
from points import BasePoints, SubPoints
from frames import Frames
from clusters import Cluster, ClusterParam
from spine import Spine
from blanpied import blanpied_clustering

class FieldOfView():
    """Class to hold and process all data within a single field of view
    
    This class is the main class the user will interact with in order to load data such as life act,
    homer centers, and points to examine for clustering. It includes multiple functions for
    plotting the data in a variety of ways for analysis. 
    
    Attributes:
        all_homer_centers (BasePoints): All homer centers found in the data.
        active_homer_centers (SubPoints): All homer centers which pass life act thresholding.
        life_act (np.ndarray): The background life act frame.
        Points (list[BasePoints]): A list containing all of the different points to analyze.
        Params (list[ClusterParam]): A list containing all of the ClusterParams used for clustering.
        clustering_results (dict[ClusterParam, list[Cluster]]): A dictionary containing the results
        from clustering in a list with the keys being the ClusterParam parameters used to find
        those clusters.


    Methods:
        __init__(): Initialize the FieldOfView class.
        locate_homer_centers(): Loads Homer centers from file for the class
        load_life_act(): Load life_act for the class.
        load_points(): Loads points for the class
        find_instance_by_label(): Find an instance of a class in a list by label.
        threshold_homers(): Apply a threshold to the homer centers based on the background life act.
        find_clusters(): Locate clusters of Points in the overall FOV using DBSCAN.
        add_params(): Process Params, and call out to self.find_clusters().
        plot_homer(): Plot the region around a homer center
        cluster_size_histogram(): Plots of histogram of the calculated size of clusters.
        cluster_size_by_distance_to_homer_center(): Plots cluster size vs. distance to nearest 
        homer center.
    """
    def __init__(self, homer_centers, life_act, nm_per_pixel=1, points=[], Params=[], 
                 threshold=0, deepd3_model_path=None, deepd3_scale=(512, 512), deepd3_pred_thresh=0.1, to_print=True):
        """
        Initialization function for FieldOfView class
        
        Args:
            homer_centers (str): path to file containing homer centers
            life_act (str): path to file containing life act movie
            nm_per_pixel (int or float): conversion ratio from nm to pixels for this FOV.
            points (list): list containing sublists of format [str label, str path, str color, 
            float time_per_frame, float Tau_D] for each set of points containing the label for those points, 
            the path to their csv file, the color, the time per frame in seconds, and their associated Tau_D.
            Params (list[ClusterParam], optional): list containing predefined ClusterParams objects 
            for DBSCAN clustering. Defaults to [].
            threshold (int or float, optional): threshold value of life_act for a homer center to be 
            included in self.active_homers. Also thresholds points. Defaults to 0.
            deepd3_model_path (str): path to deepd3 model for spine identification. Defaults to None.
            deepd3_scale ((int , int)): pixel resolution to scale to for deepd3. Defaults to (512, 512).
            deepd3_pred_thresh (float): float in range [0, 1] that is the minimum confidence threshold 
            to consider a deepd3 prediction legitimate. Defaults to 0.1.
            to_print (bool, optional): prints initialization progress. Defaults to False.
        """
        # Set scale
        self.nm_per_pixel = nm_per_pixel

        # Load Life Act
        if to_print: print("Loading Life Act...")
        self.life_act = self.load_life_act(life_act)

        # Set up thresholding
        print("Setting up Thresholding...")
        self.Spines, self.spinemap = self.deepd3_thresholding(deepd3_model_path, deepd3_scale, 
                                                              threshold, deepd3_pred_thresh)

        # Load Homer Centers
        if to_print: print("Loading Homer Centers...")   
        self.all_homer_centers = self.locate_homer_centers(homer_centers)
        self.assign_homers_to_spines()

        # Loading Points
        self.Points = []
        if not isinstance(points[0], list):
            points = [points]
        for point in points:
            if to_print: print(f"Loading {point[0]}...")
            self.Points.append(self.load_points(point[0], point[1], point[2], 
                                                point[3], self.nm_per_pixel, point[4]))
        
        # Find Clusters
        self.pseudo_pixel_size = 25 # nm
        self.Params = []
        self.clustering_results = {}
        self.add_params(Params, to_print)
        self.assign_clusters_to_spines()
        
        # Remove spines without homer or clusters
        self.filter_bad_spines(to_print=True)
    
    def locate_homer_centers(self, homer_path, plot=False):
        """
        Load Homer data from a CSV or Excel file and identify Homer centers using DBSCAN clustering.
        Remaining Homer centers are converted to pixel coordinates and returned.

        Args:
            homer_path (str): The file path to the CSV or Excel file containing Homer data. 
            The file should have the format output by ThunderSTORM, with localization data in 
            columns 2 and 3 (0-indexed).
            plot (bool, optional): If True, a scatter plot of the identified Homer centers is 
            displayed. Defaults to False.

        Returns:
            BasePoints object: all homer centers in the FOV. 

        Raises:
            FileNotFoundError: If the file specified by homer_path does not exist.
            ValueError: If the file specified by homer_path does not have the expected format.
        
        Note:
            The DBSCAN clustering uses a radius of 50 nm and requires a minimum of 5 points per 
            cluster. Clusters identified by DBSCAN are considered to be potential Homer centers, 
            and the mean position of each cluster is used as the position of the corresponding 
            Homer center.
        """
        dim = 2
        ## OLD SETTINGS ##
        synapse_size = 50  # cluster size in nm
        min_neighbours = 5  # minimum number of neighbours w/n synapse_size radius
        ## NEW SETTINGS ##
        synapse_size = 50  # cluster size in nm
        min_neighbours = 10  # minimum number of neighbours w/n synapse_size radius
        try:
            ThunderSTORM = pd.read_csv(homer_path, sep=',', skiprows=1, header=None).values
        except Exception:
            ThunderSTORM = pd.read_excel(homer_path, header=None).values
        data_Syn = ThunderSTORM[:, 2:2+dim]

        # Finding clusters
        db = DBSCAN(eps=synapse_size, min_samples=min_neighbours, metric='euclidean', n_jobs=-1).fit(data_Syn)
        Class = db.labels_
        type_ = np.array([1 if label != -1 else 0 for label in Class])

        # Make new matrix
        cluster_avgs = []
        Syn = np.column_stack((data_Syn, Class, type_))

        # Separate and plot clusters
        if plot: plt.figure()

        for k in np.unique(Class):
            if k != -1:
                cluster_points = Syn[Syn[:, dim] == k]
                cluster_avgs.append((sum(cluster_points[:, 0])/len(cluster_points[:, 0]), 
                                     sum(cluster_points[:, 1])/len(cluster_points[:, 1])))
                if plot: plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=20)
        if plot:
            plt.axis('equal')
            plt.show()

        # Convert to pixel coordinates
        homer_centers_nm = np.array(cluster_avgs)
        homer_centers_px = homer_centers_nm/self.nm_per_pixel

        return BasePoints(homer_centers_px, frames=None, nm_per_pixel=self.nm_per_pixel, marker='v', 
                          color='chartreuse', s=100, edgecolor='black', label="Homer Center")
  
    def assign_homers_to_spines(self):
        """Function to assign self.all_homer_centers to their associated spines in self.Spines."""
        indices_by_label = {}
        for i in range(len(self.all_homer_centers)):
            y, x = self.as_pixel(self.all_homer_centers[i])
            label = self.spinemap[y][x]
            if label != -1:
                if label not in indices_by_label:
                    indices_by_label[label] = []
                indices_by_label[label].append(i)
        
        for label in indices_by_label:
            self.Spines[label].set_homer(SubPoints(self.all_homer_centers, indices_by_label[label]))

    def assign_clusters_to_spines(self):
        for param in self.clustering_results:
            clusters = self.clustering_results[param]
            indices_by_label = {}
            for cluster in clusters:
                label = cluster.spine
                if label != -1:
                    if label not in indices_by_label:
                        indices_by_label[label] = []
                    indices_by_label[label].append(cluster)
            for label in indices_by_label:
                self.Spines[label].set_clusters(param, indices_by_label[label])
                
    def filter_bad_spines(self, to_print):
        print("Filtering Bad Spines...")
        good_labels = []
        for i in range(len(self.Spines)):
            if self.Spines[i].num_homers() != 0 and self.Spines[i].contains_clusters():
                good_labels.append(i)
            # else:
            #     if to_print: print(f"Bad Spine...Homers: {self.Spines[i].num_homers()}, Clusters: {self.Spines[i].contains_clusters()}")
        
        good_spines = [self.Spines[i] for i in good_labels]
        for i in range(len(good_spines)):
            good_spines[i].label = i
        if to_print: print(f"Filtered {len(self.Spines)} Spines, Finding {len(good_spines)} Good Spines")
        self.Spines = good_spines

    def load_life_act(self, life_act, print_info=False, plot_frame=False):
        """
        Function to load life_act for the class.

        Args:
            life_act (str): string path to the life act file
            print_info (bool, optional): prints information about the movie if True. 
            Default to False.
            plot_frame (bool, optional): plots the first frame of the movie if True. 
            Defaults to False.

        Returns:
            np.ndarray: the first frame of the life_act movie
        """
        try:
            with tifffile.TiffFile(life_act) as tif:
                n_frames = len(tif.pages)
                movie = np.zeros((n_frames, tif.pages[0].shape[0], 
                                    tif.pages[0].shape[1]), dtype='uint16')
                for i in range(n_frames):
                    movie[i,:,:] = tif.pages[i].asarray()
                if print_info:
                    print(f'Number of frames: {n_frames}')
                    print(f'Shape of each frame: {tif.pages[0].shape}')
                    print(f'Data type of each pixel: {tif.pages[0].dtype}')
                if plot_frame:
                    plt.figure()
                    plt.imshow(movie[0,:,:])
                    plt.show()
            life_act = movie[0]
        except:
            raise Exception(f"""Issues with path: {life_act}, could not load movie""")
        if not isinstance(life_act, np.ndarray):
            raise RuntimeError(f"life_act is of type: {type(life_act)}, must be a string to the filepath")
        return life_act
    
    def load_points(self, label, path, color, time_per_frame, nm_per_pixel, Tau_D):
        """
        Loads point data from a CSV file, converts it to pixel coordinates, and 
        creates a BasePoints object.

        Args:
            label (str): The label for these points.
            path (str): The path to the CSV file. The file should have columns 'x [nm]' and 'y [nm]'
            for the x and y coordinates, respectively, and a 'frame' column for the frame indices.
            color (str): The color for these points.
            time_per_frame (float): The temporal scale of the videos, in seconds per frame.
            nm_per_pixel (float): The spatial scale of the images, in nanometers per pixel.
            Tau_D (float): Tau_D value for these points, Defaults to -1.0.

        Returns:
            BasePoints: A BasePoints object containing the points loaded from the CSV file.

        Raises:
            FileNotFoundError: If the file specified by path does not exist.
            ValueError: If the file specified by path does not have the expected format.
        """
        if nm_per_pixel is None:
            nm_per_pixel = self.nm_per_pixel
        df = pd.read_csv(path, delimiter=',')
        x = df['x [nm]']/nm_per_pixel
        y = df['y [nm]']/nm_per_pixel
        frames = Frames(np.array(df['frame']), time_per_frame)
        pts = np.array(list(zip(x, y)))
        return BasePoints(pts, frames, nm_per_pixel, Tau_D, s=0.75, color=color, label=label)

    def find_instance_by_label(self, instances, target_label):
        """
        Function to find an instance of a class in a list by label, 
        primarily used for BasePoints class and its descendents

        Args:
            instances (list[Object]): list of objects with .label values
            target_label (str): the label of the object you are looking for

        Returns:
            Object or None: the first Object in instances where Object.label == target_label, 
            None if no object found with target_label
        """
        for instance in instances:
            try:
                if instance.label == target_label:
                    return instance
            except:
                raise Exception("instance.label failed for instance: {instance}")
        return None
    
    def threshold(self, threshold, plot=False, limits=None):
        """
        Function to apply a threshold to the homer centers and points based on the background 
        life act intensity, will set self.active_homers and self.points.
        
        Args:
            threshold (int or float): values for the minimum intensity of life_act background 
            to pass thresholding
            plot (bool, optional): will plot the pre and post thresholding background life act 
            and homer centers. Defaults to False.
            limits (list[list[int, int], list[int, int]), optional: Limits for the plot to show in 
            the format [[x_min, x_max], [y_min, y_max]]. Shows full plot if None. Defaults to None.

        Returns:
            void
        """
        try:
            threshold_map = np.array(self.life_act > threshold)
        except:
            print("thresholding failed, self.active_homers = self.all_homer_centers")
            print("points are unchanged")
            self.active_homers = self.all_homer_centers
            return
        if plot:
            if limits is None:
                limits = [[0, self.life_act.shape[1]],[0, self.life_act.shape[0]]]
            plt.figure()
            plt.imshow(self.life_act, origin='lower')
            self.all_homer_centers.add_to_plot()
            self.Points[0].add_to_plot()
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.xlim(limits[0][0], limits[0][1])
            plt.ylim(limits[1][0], limits[1][1])
            plt.show()
        hc = self.all_homer_centers
        passed_indices = np.array([i for i in range(len(hc)) if 
                                   threshold_map[min(self.life_act.shape[0]-1, int(hc[i][1])), 
                                                 min(self.life_act.shape[1]-1, int(hc[i][0]))]])
        self.active_homers = SubPoints(self.all_homer_centers, passed_indices, **hc.plot_args)
        for i in range(len(self.Points)):
            pts = self.Points[i]
            passed_indices = np.array([j for j in range(len(pts)) if 
                                       threshold_map[min(self.life_act.shape[0]-1, int(pts[j][1])), 
                                                     min(self.life_act.shape[1]-1, int(pts[j][0]))]])
            self.Points[i] = SubPoints(pts, passed_indices, **pts.plot_args)
        if plot:
            plt.figure()
            plt.imshow(self.life_act*threshold_map, origin='lower')
            self.active_homers.add_to_plot()
            self.Points[0].add_to_plot()
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.xlim(limits[0][0], limits[0][1])
            plt.ylim(limits[1][0], limits[1][1])
            plt.show()

    def threshold_homers(self, threshold, plot=False):
        """
        Function to apply a threshold to the homer centers based on the background life act 
        intensity, will set self.active_homers
        
        Args:
            threshold (int or float): values for the minimum intensity of life_act background 
            to pass thresholding
            plot (bool, optional): will plot the pre and post thresholding background life act 
            and homer centers. Defaults to False.

        Returns:
            void
        """
        try:
            threshold_map = np.array(self.life_act > threshold)
        except:
            print("thresholding failed, self.active_homers = self.all_homer_centers")
            self.active_homers = self.all_homer_centers
            return
        hc = self.all_homer_centers
        passed_indices = np.array([i for i in range(len(hc)) if threshold_map[int(hc[i][1]), 
                                                                              int(hc[i][0])]])
        self.active_homers = SubPoints(self.all_homer_centers, passed_indices, **hc.plot_args)
        
        if plot:
            plt.figure()
            plt.imshow(self.life_act, origin='lower')
            self.all_homer_centers.add_to_plot()
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.show()

            plt.figure()
            plt.imshow(self.life_act*threshold_map, origin='lower')
            self.active_homers.add_to_plot()
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.show()

    def deepd3_thresholding(self, model_path, input_shape, life_act_thresh, pred_thresh):
        """Function to locate spines using DeepD3 and Stardist

        Args:
            model_path (string): path to the deepd3 model to use.
            input_shape ((int, int)): shape to scale self.life_act to for processing.
            life_act_thresh (float): Threshold value for spines against the background.
            pred_thresh (float): Threshold value for predictions to count in range [0, 1].

        Returns:
            labels_roi (dict): A dictionary containing the roi's for each spine label
            stardist (2D array): A 2D array of labels for spines. No spine = -1, else 0, 1, 2, ...
        """
        # Load model and background
        model = load_model(model_path, compile=False)
        background = np.copy(self.life_act)
        normalized_background = 2 * (background / np.max(background)) - 1
        normalized_background = np.expand_dims(normalized_background, axis=-1)
        resized_background = np.expand_dims(tf.image.resize(normalized_background, input_shape), axis=0)
        
        # Make Predictions
        spine_predictions = model.predict(resized_background)[1]
        resized_preds = tf.image.resize(spine_predictions, self.life_act.shape).numpy().squeeze()
        bin_pred = resized_preds * (self.life_act > life_act_thresh) * (resized_preds > pred_thresh)
        normalized_predictions = normalize(bin_pred, 0, 99.8)
        
        # Use Stardist to classify predictions
        star_model = StarDist2D.from_pretrained('2D_versatile_fluo')
        starplane, _ = star_model.predict_instances(normalized_predictions, prob_thresh=0.3, nms_thresh=0.3)
        
        # Create a dictionary of pixels for each 2d stardist label
        label_dict = {}
        next_label_index = 0
        labels = []
        labels_roi = {}
        for y in range(len(starplane)):
            for x in range(len(starplane[0])):
                label = starplane[y][x]
                if label != 0:
                    # Get the spine label
                    if not label in label_dict:
                        label_dict[label] = next_label_index
                        next_label_index += 1
                    spine_label = label_dict[label]

                    # Update labels_roi
                    if spine_label in labels:
                        labels_roi[spine_label].append([x,y])
                    else:
                        labels.append(spine_label)
                        labels_roi[spine_label] = [[x,y]]
                    
                    # Update starplane
                    starplane[y][x] = spine_label
                else:
                    starplane[y][x] = -1
        Spines = []
        for i in range(len(labels)):
            label = labels[i]
            Spines.append(Spine(label, labels_roi[label], self.nm_per_pixel))
        return Spines, starplane
     
    def find_clusters(self, Param, density_factor=3.5, min_cluster_size=3, 
                      cutoff=70, nearby_radius=2500, to_print=True):
        """
        Function to locate clusters of Points in the overall FOV based on local density calculations.
        Algorithm Translated from: https://www.sciencedirect.com/science/article/pii/S1046202318304304?via%3Dihub

        Args:
            Param (ClusterParam): instance of ClusterParam to provide label for the points to cluster.
            density_factor (float, optional): factor to multiply with MMD for local density calculation.
            min_cluster_size (int, optional): minimum number of points to consider as a valid cluster.
            cutoff (float, optional): distance cutoff for clustering in nm. Defaults to 80 nm.
            to_print (bool, optional): prints when starting and how many clusters when found. Defaults to True.

        Returns:
            list[Cluster]: list of Cluster objects found based on local density calculations.
        """
        if to_print:
            print(f"Finding Clusters for: {Param}...")

        # Find points
        Points = self.find_instance_by_label(self.Points, Param.label)
        if Points is None:
            raise Exception(f"Can not find {Param.label}")
        points = np.copy(Points.points)

        synaptic_clusters, nanocluster_groups = blanpied_clustering(points, cutoff/self.nm_per_pixel, 
                                                                    density_factor, 60, min_cluster_size)

        clusters = []
        for i in range(len(synaptic_clusters)):
            nanoclusters = nanocluster_groups[i]
            for label in nanoclusters:
                cluster_indices = nanoclusters[label]
                cluster_center = points[cluster_indices[0]]
                nearby_point_indices = synaptic_clusters[i]
                spine = self.spinemap[self.as_pixel(cluster_center)]
                clusters.append(Cluster(Points, cluster_indices, self, nearby_point_indices, spine))
        
        self.clustering_results[Param] = clusters
        if to_print: print(f"Found {len(clusters)} Clusters")
        return clusters

    def add_params(self, Params=[], to_print=True):
        """
        Function to process Params, and call out to self.find_clusters()
        
        Args:
            Params (list[ClusterParam]): list of ClusterParams to feed to find_clusters
            to_print (bool, optional): prints when starting find_clusters and how many clusters when found. 
            Defaults to True.
        Returns:
            void
        """
        if not isinstance(Params, list):
            Params = [Params]
        for Param in Params:
            self.find_clusters(Param, to_print=to_print)
    
    def point_in_limits(self, point, limits):
        """
        Function to check if a point is within rectangular limits
        
        Args:
            point (list[int, int]): Point to check 
            limits (list[list[int, int], list[int, int]): Limits for the plot to show in the format
            [[x_min, x_max], [y_min, y_max]]. Shows full plot if None. Defaults to None.
        
        Returns:
            Bool: Representing wether the point is in the limits
        """
        x_bool = point[0] < limits[0][1] and point[0] > limits[0][0]
        y_bool = point[1] < limits[1][1] and point[1] > limits[1][0]
        return x_bool and y_bool

    # TODO
    def write_clusters_to_csv(self, filename, Params=[], max_dark_time=None, use_all_homers=False):
        """
        Writes clusters for different Params to CSV files

        This function takes as input a list of parameter sets, and for each parameter set,
        it writes the associated clusters to the CSV file. Each line in the file contains
        an index, the x and y coordinates of the cluster center (in nm), the number of subunits, 
        the area in square microns, and the distance to the nearest Homer center (in nm).

        Args:
            filename (str): The name of the CSV file to write to.
            Params (list, optional): A list of parameter sets. Each parameter set is associated
            with a set of clusters. Defaults to an empty list.
            max_dark_time (float, optional): The maximum dark time allowed for a cluster. 
            Clusters with a max dark time greater than this are excluded. If this argument is None, 
            all clusters are included regardless of max dark time. Defaults to None.
            use_all_homers (bool, optional): If True, distances are calculated to all Homer centers. 
            If False, distances are calculated only to active Homer centers. Defaults to False.

        Returns:
            None
        """
        if not isinstance(Params[0], ClusterParam): Params = [Params]
        lines = [['Clustering Parameters', 'Parameter Index', 'Cluster Center x (nm)', 'Cluster Center y (nm)', '# Subunits',
                      'Area (micron^2)', 'Distance to Nearest Homer Center (nm)']]
        for label in self.Spines:
            spine = self.Spines[label]
            for Param in Params:
                if not Param in spine.clusters: 
                    continue
                clusters = [cluster for cluster in spine.clusters[Param]]
                centers_px = [cluster.cluster_center for cluster in clusters]
                if use_all_homers:
                    distances = cdist(centers_px, self.all_homer_centers.points, 'euclidean')
                else:
                    distances = cdist(centers_px, self.active_homers.points, 'euclidean')
                min_distances = np.min(distances, axis=1) * self.nm_per_pixel
                filtered_clusters = []
                filtered_distances = []
                for i in range(len(clusters)):
                    if max_dark_time is not None and clusters[i].max_dark_time > max_dark_time:
                        continue
                    if min_distances[i] > 2000:
                        continue
                    filtered_clusters.append(clusters[i])
                    filtered_distances.append(min_distances[i])
                filtered_centers_px = [cluster.cluster_center for cluster in filtered_clusters]
                filtered_centers_nm = [center*self.nm_per_pixel for center in filtered_centers_px]
                average_dark_times = [cluster.average_dark_time for cluster in filtered_clusters]
                subunits = [Tau_D/dark_time for dark_time in average_dark_times]
                areas = [cluster.cluster_area() for cluster in filtered_clusters]
                for i in range(len(filtered_clusters)):
                    lines.append([str(Param), i, filtered_centers_nm[i][0], filtered_centers_nm[i][1], subunits[i], 
                                areas[i], filtered_distances[i]])
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(lines)
        print(f"{filename} created successfully!")

    def get_spine_cluster_sizes(self, spine_id, Param, max_dark_time=500, plot_sizes_over=None):
        spine = self.Spines[spine_id]
        if not Param in spine.clusters:
            return []
        clusters = [cluster for cluster in spine.clusters[Param] 
                        if cluster.max_dark_time < max_dark_time]
        if len(clusters) == 0:
            return []
        Tau_D = clusters[0].Tau_D
        average_dark_times = [cluster.average_dark_time for cluster in clusters]
        cluster_sizes = [Tau_D/dark_time for dark_time in average_dark_times]                
        if plot_sizes_over is not None:
            for i in range(len(clusters)):
                if cluster_sizes[i] > plot_sizes_over:
                    clusters[i].plot(buffer=1500, nearby_points=True)
        return cluster_sizes
    
    def get_all_cluster_sizes(self, Param, max_dark_time=500, plot_sizes_over=None):
        cluster_sizes = []
        for i in range(len(self.Spines)):
            cluster_sizes.extend(self.get_spine_cluster_sizes(i, Param, max_dark_time, plot_sizes_over))
        return cluster_sizes
    
    def get_spine_cluster_areas(self, spine_id, Param, max_dark_time=500, plot_sizes_over=None):
        spine = self.Spines[spine_id]
        if not Param in spine.clusters:
            return []
        clusters = [cluster for cluster in spine.clusters[Param] 
                        if cluster.max_dark_time < max_dark_time]
        if len(clusters) == 0:
            return []
        cluster_areas = [cluster.cluster_area() for cluster in clusters]
        if plot_sizes_over is not None:
            for i in range(len(clusters)):
                if cluster_areas[i] > plot_sizes_over:
                    print("Dark Time: " + str(clusters[i].max_dark_time))
                    clusters[i].plot(buffer=1500, nearby_points=True)
        return cluster_areas
    
    def get_all_cluster_areas(self, Param, max_dark_time=500, plot_sizes_over=None):
        cluster_areas = []
        for i in range(len(self.Spines)):
            cluster_areas.extend(self.get_spine_cluster_areas(i, Param, max_dark_time, plot_sizes_over))
        return cluster_areas
    
    def get_spine_cluster_densities(self, spine_id, Param, max_dark_time=500, plot_sizes_over=None):
        sizes = self.get_spine_cluster_sizes(spine_id, Param, max_dark_time, plot_sizes_over)
        areas = self.get_spine_cluster_areas(spine_id, Param, max_dark_time, plot_sizes_over)
        return [sizes[i]/areas[i] for i in range(len(sizes))]
    
    def get_all_cluster_densities(self, Param, max_dark_time=500, plot_sizes_over=None):
        sizes = self.get_all_cluster_sizes(Param, max_dark_time, plot_sizes_over)
        areas = self.get_all_cluster_areas(Param, max_dark_time, plot_sizes_over)
        return [sizes[i]/areas[i] for i in range(len(sizes))]
    
    def get_spine_distances_to_homer(self, spine_id, Param, max_dark_time=None, use_all_homers=False):
        spine = self.Spines[spine_id]
        if not Param in spine.clusters:
            return []
        if max_dark_time is not None:
            cluster_centers = np.array([cluster.cluster_center for cluster in spine.clusters[Param]
                                        if cluster.max_dark_time < max_dark_time])
        else:
            cluster_centers = np.array([cluster.cluster_center for cluster in spine.clusters[Param]])
        # print(f"Spine {spine_id} Homers: {spine.homers.points}")
        if len(cluster_centers) == 0:
            return []
        if use_all_homers:
            distances = cdist(cluster_centers, self.all_homer_centers.points, 'euclidean')
        else:
            distances = cdist(cluster_centers, spine.homers.points, 'euclidean')
        min_distances = np.min(distances, axis=1) * self.nm_per_pixel
        return min_distances
    
    def get_all_distances_to_homer(self, Param, max_dark_time=500, use_all_homers=False):
        min_distances = []
        for i in range(len(self.Spines)):
            min_distances.extend(self.get_spine_distances_to_homer(i, Param, max_dark_time, use_all_homers))
        return min_distances

    def get_spinemap(self):
        return self.spinemap

    def get_life_act(self):
        return self.life_act
    
    def distance_squared(self, p1, p2):
        return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

    def as_pixel(self, point):
        """Function to convert a float point to int pixels.

        Args:
            point ((float, float)): the point to convert.

        Returns:
            tuple(int, int): (y, x) pixel coordinates of the point.
        """
        return (min(self.life_act.shape[0]-1, int(point[1])), 
                min(self.life_act.shape[1]-1, int(point[0])))
    