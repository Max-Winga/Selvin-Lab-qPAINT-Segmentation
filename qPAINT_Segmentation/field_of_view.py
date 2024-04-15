import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import tifffile
import pandas as pd
from sklearn.cluster import DBSCAN
import csv
import json
from tqdm import tqdm

from points import BasePoints, SubPoints
from frames import Frames
from clusters import Cluster
from cluster_algorithms import ClusteringAlgorithm
from spine import Spine


class FieldOfView():
    """Class to hold and process all data within a single field of view
    
    This class is the main class the user will interact with in order to load data such as life act,
    homer centers, and points to examine for clustering. It includes multiple functions for
    plotting the data in a variety of ways for analysis. Clustering of points is done through an 
    adaptation of the DBSCAN algorithm, which was first introduced by Dr. Thomas Blanpied.
    
    Attributes:
        nm_per_pixel (float): Scale of the image background.
        homer_centers (BasePoints): All homer centers found in the data.
        life_act (np.ndarray): The background life act image.
        Points (list[BasePoints]): A list containing all of the different points to analyze.
        Params (list[ClusterParam]): A list containing all of the ClusterParams used for clustering.
        clustering_results (dict[ClusterParam : list[Cluster]]): A dictionary containing the results
        from clustering in a list with the keys being the ClusterParam parameters used to find
        those clusters.
        Spines (list[Spine]): list of identified Spine objects.
        spinemap (2D array): A 2D array of spine labels. No spine = -1, else 0, 1, 2, ...


    Methods:
        __init__(): Initialize the FieldOfView class.
        locate_homer_centers(): Loads Homer centers from file for the class.
        assign_homers_to_spines(): Assigns homer centers to spines.
        assign_points_to_spines(): Assigns points to spines.
        assign_clusters_to_spines(): Assigns clusters to spines.
        filter_bad_spines(): Removes spines without Homers and clusters.
        set_cluster_numbers(): Reassigns spine and cluster indices within Clusters.
        load_life_act(): Load life_act for the class.
        load_points(): Loads points for the class
        find_instance_by_label(): Find an instance of a class in a list by label.
        threshold(): Filters out background points based on the life act intensity.
        create_spines_from_starplane(): Creates Spine instances from the starplane and labels_roi.
        find_clusters(): Locate clusters of Points in the overall FOV using DBSCAN.
        add_params(): Process Params, and call out to self.find_clusters().
        point_in_limits(): Checks whether a point is within rectangular limits.
        write_clusters_to_csv(): Writes cluster data to CSV.
        get_spine_cluster_sizes(): Retrieves the clusters sizes for a given spine.
        get_all_cluster_sizes(): Retrieves all cluster sizes.
        get_spine_cluster_areas(): Retrieves the clusters areas for a given spine.
        get_all_cluster_areas(): Retrieves all cluster areas.
        get_spine_cluster_densities(): Retrieves the clusters densities for a given spine.
        get_all_cluster_densities(): Retrieves all cluster densities.
        get_spine_distances_to_homer(): Retrieves the cluster distances to Homer for a given spine.
        get_all_distances_to_homer(): Retrieves all cluster distances to Homer.
        get_spinemap(): Getter for spinemap.
        get_life_act(): Getter for life_act.
        get_homers(): Getter for homer_centers.
        as_pixel(): Converts float coordinates to integer pixel coordinates.
    """
    def __init__(self, homer_centers, life_act, starplane_file, labels_roi_file, nm_per_pixel=1, 
                 points=[], cluster_algorithms=[], to_print=True, filter_spines=True, multithreading=-1):
        """
        Initialization function for FieldOfView class
        
        Args:
            homer_centers (str): path to file containing homer centers
            life_act (str): path to file containing life act movie
            starplane_file (str): path to the .npy file containing the starplane data
            labels_roi_file (str): path to the .json file containing the labels_roi data
            nm_per_pixel (int or float): conversion ratio from nm to pixels for this FOV.
            points (list): list containing sublists of format [str label, str path, str color, 
            float time_per_frame, float Tau_D] for each set of points containing the label for those points, 
            the path to their csv file, the color, the time per frame in seconds, and their associated Tau_D.
            cluster_algorithms (list(ClusterAlgorithm)): A list of ClusterAlgorithm's to run on the data. Defaults to [].
            to_print (bool, optional): prints initialization progress. Defaults to False.
            filter_spines (bool, optional): removes spines without Homers or clusters. Defaults to True.
            multithreading (int, optional): Enables multithreading. SET TO 1 FOR M1/2 MACS. Defaults to -1 (all cores).
        """
        # Set values
        self.nm_per_pixel = nm_per_pixel
        self.multithreading = multithreading

        # Load Life Act
        if to_print: print("Loading Life Act...")
        self.life_act = self.load_life_act(life_act)

        # Load starplane and labels_roi from files
        if to_print: print("Loading Starplane and Labels ROI...")
        starplane = np.load(starplane_file)
        with open(labels_roi_file, 'r') as f:
            labels_roi = json.load(f)

        # Create spines from starplane
        print("Creating Spines from Starplane...")
        self.Spines, self.spinemap = self.create_spines_from_starplane(starplane, labels_roi)

        # Load Homer Centers
        if to_print: print("Loading Homer Centers...")   
        self.homer_centers = self.locate_homer_centers(homer_centers)
        self.assign_homers_to_spines()

        # Loading Points
        self.Points = []
        if not isinstance(points[0], list):
            points = [points]
        for point in points:
            if to_print: print(f"Loading {point[0]}...")
            self.Points.append(self.load_points(point[0], point[1], point[2], 
                                                point[3], self.nm_per_pixel, 
                                                point[4]))
        if to_print: print("Assigning Points to Spines...")
        self.assign_points_to_spines()
        
        # Find Clusters
        self.ClusterAlgs = cluster_algorithms
        self.find_clusters(cluster_algorithms, to_print)
        
        # Remove spines without homer or clusters
        if filter_spines: self.filter_bad_spines(to_print=True)
        
        # Make sure clusters contain the right indices
        self.set_cluster_numbers()
    
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
        """Function to assign self.homer_centers to their associated spines in self.Spines."""
        indices_by_label = {}
        for i in range(len(self.homer_centers)):
            y, x = self.as_pixel(self.homer_centers[i])
            label = self.spinemap[y][x]
            if label != -1:
                if label not in indices_by_label:
                    indices_by_label[label] = []
                indices_by_label[label].append(i)
        
        for label in indices_by_label:
            self.Spines[label].set_homer(SubPoints(self.homer_centers, indices_by_label[label]))

    def assign_points_to_spines(self):
        """Function to assign points in spinal regions to their respective spines"""
        for points in self.Points:
            label = points.label
            spines = np.array([self.spinemap[self.as_pixel(pt)] for pt in points])
            for i in range(np.max(self.spinemap) + 1):
                self.Spines[i].points[label] = Cluster(points, np.where(spines == i), fov=self, spine=i, label=label)
                
    def filter_bad_spines(self, to_print=False):
        """A function to filter out spines without homer centers or clusters"""
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
    
    def set_cluster_numbers(self):
        """A function to reassign the spine and cluster indices within the clusters"""
        for i in range(len(self.Spines)):
            for cluster_alg in self.ClusterAlgs:
                for j in range(len(self.Spines[i].clusters[cluster_alg])):
                    cluster = self.Spines[i].clusters[cluster_alg][j]
                    cluster.spine = i
                    cluster.cluster_number = j

    def load_life_act(self, life_act, print_info=False, plot_frame=False):
        """
        Function to load life_act for the class from a file.

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
    
    def load_points(self, label, path, color, time_per_frame, nm_per_pixel, Tau_D, life_act_thresh=0):
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
            Tau_D (float): Tau_D (average dark time) value for these points.
            life_act_thresh (float, optional): Background threshold to be considered a point. Defaults to 0.

        Returns:
            BasePoints: A BasePoints object containing the points loaded from the CSV file.
                        COORDINATES ARE IN PIXEL SCALE!!!

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
        pts = np.array([pt for pt in list(zip(x, y))])
        full_basepoints = BasePoints(pts, frames, nm_per_pixel, Tau_D, s=0.75, color=color, label=label)
        if life_act_thresh != 0:
            filter_indices = [self.life_act[self.as_pixel(pt)] > life_act_thresh for pt in pts]
            filtered_basepoints = SubPoints(full_basepoints, filter_indices, label=label)
            return filtered_basepoints
        else:
            return full_basepoints

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

    def create_spines_from_starplane(self, starplane, labels_roi):
        """Function to create Spine instances from the starplane and labels_roi.

        Args:
            starplane (np.ndarray): 2D array of spine labels.
            labels_roi (dict): Dictionary mapping spine labels to ROI coordinates.

        Returns:
            Spines (list[Spine]): A list containing Spine instances.
            starplane (np.ndarray): The starplane array.
        """
        Spines = []
        for label, roi in labels_roi.items():
            Spines.append(Spine(int(label), roi, self.nm_per_pixel))
        return Spines, starplane
    
    def find_clusters(self, clustering_algorithms, to_print=True):
        """
        Function to locate clusters of Points in the overall FOV based on local density calculations.

        Args:
            clustering_algorithms (list(ClusterAlgorithm)): List of ClusteringAlgorithm's to apply.
            to_print (bool, optional): prints when starting and how many clusters when found. Defaults to True.

        Returns:
            list[Cluster]: list of Cluster objects found based on local density calculations.
        """
        if to_print:
            print("Starting clustering...")

        # Create a progress bar
        progress_bar = tqdm(total=len(self.Spines) * len(clustering_algorithms), unit='spine', desc='Clustering progress')

        # Iterate through each spine:
        for spine in self.Spines:
            points = spine.points
            for ClusterAlg in clustering_algorithms:
                clusters = ClusterAlg(points[ClusterAlg.target_points])
                spine.set_clusters(ClusterAlg, clusters)
                progress_bar.update(1)  # Update the progress bar after each clustering operation

        # Close the progress bar
        progress_bar.close()

        if to_print:
            print("Clustering completed.")

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

    def write_clusters_to_csv(self, filename, clustering_algorithms=[], max_dark_time=None, use_all_homers=False):
        """
        Writes clusters for different clustering algorithms to CSV files.

        This function takes as input a list of clustering algorithms, and for each algorithm,
        it writes the associated clusters to the CSV file. Each line in the file contains the algorithm label,
        the spine label, the index, the x and y coordinates of the cluster center (in nm), the number of subunits, 
        the area in square microns, and the distance to the nearest Homer center (in nm).

        Args:
            filename (str): The name of the CSV file to write to.
            clustering_algorithms (list, optional): A list of ClusteringAlgorithm objects. Each algorithm is associated
                with a set of clusters. Defaults to an empty list.
            max_dark_time (float, optional): The maximum dark time (in seconds) allowed for a cluster. 
                Clusters with a max dark time greater than this are excluded. If this argument is None, 
                all clusters are included regardless of max dark time. Defaults to None.
            use_all_homers (bool, optional): If True, distances are calculated to all Homer centers. 
                If False, distances are calculated only to the spine Homers. Defaults to False.

        Returns:
            None
        """
        if isinstance(clustering_algorithms, ClusteringAlgorithm):
            clustering_algorithms = [clustering_algorithms]
        
        lines = [['Clustering Algorithm Label', 'Spine Label', 'Cluster Index', 'Cluster Center x (nm)', 'Cluster Center y (nm)', 
                '# Subunits', 'Area (micron^2)', 'Distance to Nearest Homer Center (nm)', 'Max Dark Time (s)']]
        
        for spine_idx in range(len(self.Spines)):
            spine = self.Spines[spine_idx]
            for alg in clustering_algorithms:
                if alg not in spine.clusters:
                    continue
                clusters = [cluster for cluster in spine.clusters[alg] 
                            if cluster.max_dark_time < max_dark_time]
                cluster_centers_nm = [cluster.cluster_center * self.nm_per_pixel for cluster in clusters]
                min_distances = self.get_spine_distances_to_homer(spine_idx, alg, max_dark_time, use_all_homers)
                subunits = self.get_spine_cluster_sizes(spine_idx, alg, max_dark_time)
                areas = self.get_spine_cluster_areas(spine_idx, alg, max_dark_time)
                for i in range(len(clusters)):
                    lines.append([str(alg), spine_idx, i, cluster_centers_nm[i][0], cluster_centers_nm[i][1], subunits[i], 
                                areas[i], min_distances[i], clusters[i].max_dark_time])
        
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(lines)
        
        print(f"{filename} created successfully!")
    
    def write_spines_to_csv(self, filename, points, max_dark_time=None):
        """
        Writes spines as clusters for different points to a CSV file.

        This function takes as input a list of points (i.e. ['GluA1', 'GluA2']), and for each set,
        it writes the characteristics of the event to the CSV file. Each line in the file contains the point label,
        the spine label, the x and y coordinates of the cluster center (in nm), the number of subunits, 
        the area in square microns, and the maximum dark time on the trace of the reading.

        Args:
            filename (str): The name of the CSV file to write to.
            points (list): A list of point labels to print.
            max_dark_time (float, optional): The maximum dark time (in seconds) allowed for a cluster. 
                Clusters with a max dark time greater than this are excluded. If this argument is None, 
                all clusters are included regardless of max dark time. Defaults to None.

        Returns:
            None
        """
        lines = [['Points Label', 'Spine Label', 'Cluster Center x (nm)', 'Cluster Center y (nm)', 
                '# Subunits', 'Area (micron^2)', 'Max Dark Time (s)']]
        
        for point_label in points:
            for spine_idx in range(len(self.Spines)):
                spine = self.Spines[spine_idx]
                if point_label not in spine.points:
                    continue
                
                points = spine.points[point_label]
                mdt = points.max_dark_time
                if (max_dark_time and mdt > max_dark_time) or points.average_dark_time <= 0:
                    continue
                
                center = points.cluster_center * self.nm_per_pixel
                subunits = points.Tau_D / points.average_dark_time
                area = points.cluster_area()
                
                lines.append([point_label, spine_idx, center[0], center[1], subunits, area, mdt])
        
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(lines)
        
        print(f"{filename} created successfully!")
    
    def get_spine_cluster_sizes(self, spine_id, alg, max_dark_time=500, plot_sizes_over=None):
        """Function to retrieve the size (in # subunits) of each cluster in the spine.
        
        Parameters:
            spine_id (int): The index of the spine to analyze.
            alg (ClusteringAlgorithm): The clustering algorithm indicating the clusters to retrieve.
            max_dark_time (int, optional): Maximum dark time (in seconds) that a cluster is 
                                        allowed to have without being filtered out. Defaults to 500.
            plot_sizes_over (int, optional): Size threshold (in micron^2) to plot clusters for 
                                            troubleshooting. Defaults to None.
        
        Returns:
            sizes (list[float]): An ordered list of sizes (in # subunits) from each cluster.
        """
        spine = self.Spines[spine_id]
        if alg not in spine.clusters:
            return []
        
        clusters = [cluster for cluster in spine.clusters[alg] 
                    if cluster.max_dark_time < max_dark_time]
        
        if len(clusters) == 0:
            return []
        
        Tau_D = clusters[0].Tau_D
        average_dark_times = [cluster.average_dark_time for cluster in clusters]
        cluster_sizes = [Tau_D / dark_time for dark_time in average_dark_times]
        
        if plot_sizes_over is not None:
            for i in range(len(clusters)):
                if cluster_sizes[i] > plot_sizes_over:
                    clusters[i].plot(buffer=1500, nearby_points=True)
        
        return cluster_sizes
    
    def get_all_cluster_sizes(self, alg, max_dark_time=500, plot_sizes_over=None):
        """Function to retrieve the size (in # subunits) of each cluster in each spine.
        
        Parameters:
            alg (ClusteringAlgorithm): The clustering algorithm indicating the clusters to retrieve.
            max_dark_time (int, optional): Maximum dark time (in seconds) that a cluster is allowed 
                                        to have without being filtered out. Defaults to 500.
            plot_sizes_over (int, optional): Size threshold (in micron^2) to plot clusters for 
                                            troubleshooting. Defaults to None.
        
        Returns:
            sizes (list[float]): An ordered list of sizes from each cluster of every spine.
        """
        cluster_sizes = []
        for i in range(len(self.Spines)):
            cluster_sizes.extend(self.get_spine_cluster_sizes(i, alg, max_dark_time, plot_sizes_over))
        
        return cluster_sizes
    
    def get_spine_cluster_areas(self, spine_id, alg, max_dark_time=500, plot_sizes_over=None):
        """Function to retrieve the area of each cluster in the spine (in micron^2).
        
        Parameters:
            spine_id (int): The index of the spine to analyze.
            alg (ClusteringAlgorithm): The clustering algorithm indicating the clusters to retrieve.
            max_dark_time (int, optional): Maximum dark time (in seconds) that a cluster is allowed 
                                        to have without being filtered out. Defaults to 500.
            plot_sizes_over (int, optional): Size threshold (in micron^2) to plot clusters for 
                                            troubleshooting. Defaults to None.
        
        Returns:
            areas (list[float]): An ordered list of areas from each cluster.
        """
        spine = self.Spines[spine_id]
        if alg not in spine.clusters:
            return []
        
        clusters = [cluster for cluster in spine.clusters[alg] 
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

    def get_all_cluster_areas(self, alg, max_dark_time=500, plot_sizes_over=None):
        """Function to retrieve the area of each cluster in each spine (in micron^2).
        
        Parameters:
            alg (ClusteringAlgorithm): The clustering algorithm indicating the clusters to retrieve.
            max_dark_time (int, optional): Maximum dark time (in seconds) that a cluster is allowed 
                                        to have without being filtered out. Defaults to 500.
            plot_sizes_over (int, optional): Size threshold (in micron^2) to plot clusters for 
                                            troubleshooting. Defaults to None.
        
        Returns:
            areas (list[float]): An ordered list of areas from each cluster of every spine.
        """
        cluster_areas = []
        for i in range(len(self.Spines)):
            cluster_areas.extend(self.get_spine_cluster_areas(i, alg, max_dark_time, plot_sizes_over))
        
        return cluster_areas
    
    def get_spine_cluster_densities(self, spine_id, alg, max_dark_time=500, plot_sizes_over=None):
        """Function to retrieve the density of each cluster in the spine (in # subunits/micron^2).
        
        Parameters:
            spine_id (int): The index of the spine to analyze.
            alg (ClusteringAlgorithm): The clustering algorithm indicating the clusters to retrieve.
            max_dark_time (int, optional): Maximum dark time (in seconds) that a cluster is allowed 
                                        to have without being filtered out. Defaults to 500.
            plot_sizes_over (int, optional): Size threshold (in micron^2) to plot clusters for 
                                            troubleshooting. Defaults to None.
        
        Returns:
            densities (list[float]): An ordered list of densities from each cluster.
        """
        sizes = self.get_spine_cluster_sizes(spine_id, alg, max_dark_time, plot_sizes_over)
        areas = self.get_spine_cluster_areas(spine_id, alg, max_dark_time, plot_sizes_over)
        
        return [sizes[i] / areas[i] for i in range(len(sizes))]

    def get_all_cluster_densities(self, alg, max_dark_time=500, plot_sizes_over=None):
        """Function to retrieve the density of each cluster in each spine (in # subunits/micron^2).
        
        Parameters:
            alg (ClusteringAlgorithm): The clustering algorithm indicating the clusters to retrieve.
            max_dark_time (int, optional): Maximum dark time (in seconds) that a cluster is allowed 
                                        to have without being filtered out. Defaults to 500.
            plot_sizes_over (int, optional): Size threshold (in micron^2) to plot clusters for 
                                            troubleshooting. Defaults to None.
        
        Returns:
            densities (list[float]): An ordered list of densities from each cluster of every spine.
        """
        sizes = self.get_all_cluster_sizes(alg, max_dark_time, plot_sizes_over)
        areas = self.get_all_cluster_areas(alg, max_dark_time, plot_sizes_over)
        
        return [sizes[i] / areas[i] for i in range(len(sizes))]

    def get_spine_distances_to_homer(self, spine_id, alg, max_dark_time=500, use_all_homers=False):
        """Function to retrieve the minimum distances from each cluster in a spine to the Homer center.
        
        Parameters:
            spine_id (int): The index of the spine to analyze.
            alg (ClusteringAlgorithm): The clustering algorithm indicating the clusters to retrieve.
            max_dark_time (int, optional): Maximum dark time (in seconds) that a cluster is allowed 
                                        to have without being filtered out. Defaults to 500.
            use_all_homers (bool, optional): Whether to use all Homer centers or just those attached to
                                            the spine. Defaults to False.
        
        Returns:
            min_distances (arraylike[float]): An ordered list of minimum distances (in nm) from each 
                                            cluster to the nearest Homer Center.
        """
        spine = self.Spines[spine_id]
        if alg not in spine.clusters:
            return []
        
        if max_dark_time is not None:
            cluster_centers = np.array([cluster.cluster_center for cluster in spine.clusters[alg]
                                        if cluster.max_dark_time < max_dark_time])
        else:
            cluster_centers = np.array([cluster.cluster_center for cluster in spine.clusters[alg]])
        
        if len(cluster_centers) == 0:
            return []
        
        if use_all_homers:
            distances = cdist(cluster_centers, self.homer_centers.points, 'euclidean')
        else:
            distances = cdist(cluster_centers, spine.homers.points, 'euclidean')
        
        min_distances = np.min(distances, axis=1) * self.nm_per_pixel
        return min_distances

    def get_all_distances_to_homer(self, alg, max_dark_time=500, use_all_homers=False):
        """Function to retrieve the minimum distances from each cluster in each spine to the Homer center.
        
        Parameters:
            alg (ClusteringAlgorithm): The clustering algorithm indicating the clusters to retrieve.
            max_dark_time (int, optional): Maximum dark time (in seconds) that a cluster is allowed 
                                        to have without being filtered out. Defaults to 500.
            use_all_homers (bool, optional): Whether to use all Homer centers or just those attached to
                                            spines. Defaults to False.
        
        Returns:
            min_distances (list[float]): An ordered list of minimum distances (in nm) from each  
                                        cluster to the nearest Homer Center.
        """
        min_distances = []
        for i in range(len(self.Spines)):
            min_distances.extend(self.get_spine_distances_to_homer(i, alg, max_dark_time, use_all_homers))
        
        return min_distances
    
    def get_spinemap(self):
        return self.spinemap

    def get_life_act(self):
        return self.life_act

    def get_homers(self):
        return self.homer_centers
    
    def as_pixel(self, point):
        """Function to convert a float point to int pixels.

        Args:
            point ((float, float)): the point to convert.

        Returns:
            tuple(int, int): (y, x) pixel coordinates of the point.
        """
        return (min(self.life_act.shape[0]-1, int(point[1])), 
                min(self.life_act.shape[1]-1, int(point[0])))
    