import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import tifffile
import pandas as pd
from sklearn.cluster import DBSCAN
import warnings
import csv

from plot_helpers import plot_scale_bar, PlotColors
from points import BasePoints, SubPoints
from frames import Frames
from clusters import Cluster, ClusterParam

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
        cluster_size_hisogram(): Plots of histogram of the calculated size of clusters.
        cluster_size_by_distance_to_homer_center(): Plots cluster size vs. distance to nearest 
        homer center.
    """
    def __init__(self, homer_centers, life_act, nm_per_pixel=1, points=[], 
                 Params=[], threshold=0, to_print=True):
        """
        Initialization function for FieldOfView class
        
        Args:
            homer_centers (str or BasePoints): either path to file containing homer centers or 
            BasePoints class with homer centers
            life_act (str or array-like): either path to file containing life act movie or 
            already loaded life_act FOV
            nm_per_pixel (int or float): conversion ratio from nm to pixels for this FOV
            points (list[list[str label, str path, str color, float time_per_frame]): 
            list containing sublists for each set of points containing the label for those
            points, the path to their csv file, the color, and the time per frame in seconds.
            Params (list[ClusterParam], optional): list containing predefined ClusterParams objects 
            for DBSCAN clustering. Defaults to [].
            threshold (int or float, optional): threshold value of life_act for a homer center to be 
            included in self.active_homers. Also thresholds points. Defaults to 0.
            to_print (bool, optional): prints initialization progress. Defaults to False.
        """
        self.nm_per_pixel = nm_per_pixel
        if isinstance(homer_centers, str):
            if to_print: print("Loading Homer Centers...")
            self.locate_homer_centers(homer_centers)
        else:
            if not isinstance(homer_centers, BasePoints):
                raise RuntimeError("homer_centers must be either a path or BasePoints objects")
            self.all_homer_centers = homer_centers
        if to_print: print("Loading Life Act...")
        self.life_act = self.load_life_act(life_act)
        self.Points = []
        if not isinstance(points[0], list):
            points = [points]
        for point in points:
            if to_print: print(f"Loading {point[0]}...")
            self.Points.append(self.load_points(point[0], point[1], point[2], 
                                                point[3], self.nm_per_pixel))
        self.threshold(threshold)
        self.Params = []
        self.clustering_results = {}
        self.add_params(Params, to_print)
    
    def locate_homer_centers(self, homer_path, plot=False):
        """
        Load Homer data from a CSV or Excel file and identify Homer centers using DBSCAN clustering.
        
        Homer centers are converted to pixel coordinates and stored as a BasePoint object in 
        self.all_homer_centers.

        Args:
            homer_path (str): The file path to the CSV or Excel file containing Homer data. 
            The file should have the format output by ThunderSTORM, with localization data in 
            columns 2 and 3 (0-indexed).
            plot (bool, optional): If True, a scatter plot of the identified Homer centers is 
            displayed. Defaults to False.

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
        db = DBSCAN(eps=synapse_size, min_samples=min_neighbours, metric='euclidean').fit(data_Syn)
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

        self.all_homer_centers = BasePoints(homer_centers_px, frames=None, 
                                            nm_per_pixel=self.nm_per_pixel, 
                                            marker='v', color='chartreuse', 
                                            s=100, edgecolor='black', 
                                            label="Homer Center")
  
    def load_life_act(self, life_act, print_info=False, plot_frame=False):
        """
        Function to load life_act for the class.

        Args:
            life_act (str or np.ndarray): should be either the string path to the life act file, 
            or life_act frame.
            print_info (bool, optional): prints information about the movie if True. 
            Default to False.
            plot_frame (bool, optional): plots the first frame of the movie if True. 
            Defaults to False.

        Returns:
            np.ndarray: the first frame of the life_act movie
        """
        if isinstance(life_act, str):
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
            warnings.warning(f"life_act is of type: {type(life_act)}")
        return life_act
    
    def load_points(self, label, path, color, time_per_frame, nm_per_pixel):
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
        return BasePoints(pts, frames, nm_per_pixel, s=0.75, color=color, label=label)

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

    def find_clusters(self, Param, nearby_radius=2500, to_print=True):
        """
        Function to locate clusters of Points in the overall FOV using DBSCAN with 
        parameters eps and min_samples (from Param)
        
        Args:
            Param (ClusterParam): instance of ClusterParam to provide eps and min_samples for the
            DBSCAN clustering as well as the points to cluster
            nearby_radius (int or float, optional): number in nm representing the radius around the 
            cluster center to consider as nearby points for plotting close 
            (faster than plotting all points every time)
            to_print (bool, optional): prints when starting and how many clusters when found. 
            Defaults to True.

        Returns:
            list[Cluster]: list of Cluster objects found from the DBSCAN
        """
        if to_print: print(f"Finding Clusters for: {Param}")
        Points = self.find_instance_by_label(self.Points, Param.label)
        if Points is None:
            raise Exception(f"Can not find {Param.label}")
        if Param not in self.Params:
            self.Params.append(Param)
        eps = Param.eps / Points.nm_per_pixel
        min_samples = Param.min_samples
        clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(Points.points)
        labels = clustering.labels_
        indices = np.arange(0, len(Points))
        clusters = []
        cluster_length = np.max(labels) + 1
        for i in range(cluster_length):
            cluster_indices = indices[labels == i]
            this_cluster = Cluster(Points, cluster_indices, fov=self, s=0.75, 
                                   color='aqua', label=f'Cluster {i}')
            if this_cluster.average_dark_time == -1:
                ## Accounting for clusters that are too small
                cluster_length += -1
                continue
            clusters.append(this_cluster)
        cluster_centers = [cluster.cluster_center for cluster in clusters]
        kdtree = KDTree(Points.points)
        nearby_point_indices = kdtree.query_ball_point(cluster_centers, 
                                                       nearby_radius/Points.nm_per_pixel, 
                                                       workers=-1)
        for i in range(cluster_length):
            clusters[i].nearby_points = SubPoints(Points, nearby_point_indices[i], 
                                                  label="Nearby " + Points.label)
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
            self.find_clusters(Param, to_print)
    
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
    
    def plot_region(self, limits=None, Params=[], circle_radii=[], show_points=[], dpi=150, 
                   life_act=True, cluster_centers=False, homers=True, homer_size=100, scale_bar=True,
                   ticks=True, legend=True, background_points_colors=['white', 'cyan', 'lime'],
                   cluster_colors=['red', 'orange', 'yellow', 'purple'], background_cmap='bone',
                   uniform_cluster_colors=True, point_size=0.75, circle_colors=['white', 'blue'],
                   homer_color='chartreuse', box_limits=None, box_color='white'):
        """
        Function to plot the a region of the overall field of view.
        
        Args:
            limits (list[list[int, int], list[int, int]), optional: Limits for the plot to show in 
            the format [[x_min, x_max], [y_min, y_max]]. Shows full plot if None. Defaults to None.
            Params (list[ClusterParam], optional): List of ClusterParams to view clusters from.
            Defaults to [].
            circle_radii (list[int or float], optional): Radii (nm) to draw circles around central 
            homer. Defaults to [].
            show_points (list[str], optional): List of labels of points in self.Points to display.
            Defaults to [].
            dpi (int, optional): DPI of image to be passed to plt.figure(dpi). Defaults to 150.
            life_act (bool, optional): Display background life act. Defaults to True.
            cluster_centers (bool, optional): Mark centers of each cluster. Defaults to False.
            homers (bool, optional): Show homer centers. Defaults to True.
            homer_size (int): Size to make the homer points. Defaults to 100.
            scale_bar (bool, optional): Show scale bar. Defaults to True.
            ticks (bool, optional): Show plot ticks (in pixels, NOT nm). Defaults to True.
            legend (bool, optional): Show legend. Defaults to True.
            background_points_colors (list[str], optional): List of matplotlib colors to use for 
            show_points. Defaults to ['white', 'cyan', 'lime'].
            cluster_colors (list[str], optional): List of matplotlib colors to use for clusters.
            Defaults to ['red', 'orange', 'yellow', 'purple'].
            uniform_cluster_colors (bool, optional): When plotting just one param, choose whether 
            all clusters are the same color or each cluster is different. Defaults to True.
            background_cmap (str): Color map for the background Life Act. Defaults to 'bone'
            point_size (float): Size of the points on the plot. Defaults to 0.75.
            circle_colors (list[str]): Colors for the circles in order of the circle_radii. Defaults
            to ['white', 'blue'].
            homer_color (str): Color for Homer centers. Defaults to 'chartreuse'.
            box_limits (list[list[int, int], list[int, int]): Limits for a box to show on the plot
            in the format [[x_min, x_max], [y_min, y_max]]. No box if None. Defaults to None.
            box_color (std): Color to make the box object defined by box_limits
            
        
        Returns:
            void: Just shows the plot.
        """
        if limits is None:
            limits = [[0, self.life_act.shape[1]],[0, self.life_act.shape[0]]]
        background_points_colors = PlotColors(background_points_colors)
        cluster_colors = PlotColors(cluster_colors)
        plt.figure(dpi=dpi)
        # Plot Life Act Background
        if life_act:
            try:
                plt.imshow(self.life_act, cmap=background_cmap, origin='lower')
            except:
                warnings.warning("Cannot show Life_Act")
        
        # Set Plot Ranges
        plt.xlim(limits[0][0], limits[0][1])
        plt.ylim(limits[1][0], limits[1][1])
        
        # Plot Background Points
        if not isinstance(show_points, list):
            show_points = [show_points]
        for i in range(len(show_points)):
            Points = self.find_instance_by_label(self.Points, show_points[i])
            nearby_point_indices = [i for i in range(len(Points)) 
                                    if self.point_in_limits(Points.points[i], limits)]
            SubPoints(Points, nearby_point_indices, s=point_size, 
                      color=background_points_colors.get_next_color()).add_to_plot()
        
        # Plot Clusters
        if not isinstance(Params, list):
            Params = [Params]
        if len(Params) == 1 and not uniform_cluster_colors:
            try:
                clusters = self.clustering_results[Params[0]]
            except:
                print(f"{Params[0]} has not been run yet, running find_clusters...")
                self.find_clusters(Params[0])
                clusters = self.clustering_results[Params[0]]
            cluster_centers = [cluster.cluster_center for cluster in clusters]
            nearby_cluster_indices = [i for i in range(len(cluster_centers)) 
                                    if self.point_in_limits(cluster_centers[i], limits)]
            for i in nearby_cluster_indices:
                    clusters[i].add_to_plot(color=None, s=point_size)
        else:
            for i in range(len(Params)):
                Param = Params[i]
                try:
                    clusters = self.clustering_results[Param]
                except:
                    print(f"{Param} has not been run yet, running find_clusters...")
                    self.find_clusters(Param)
                    clusters = self.clustering_results[Param]
                cluster_centers = [cluster.cluster_center for cluster in clusters]
                nearby_cluster_indices = [i for i in range(len(cluster_centers)) 
                                    if self.point_in_limits(cluster_centers[i], limits)]
                cluster_level_indices = []
                for j in nearby_cluster_indices:
                    cluster_level_indices.extend(clusters[j].indices)
                Points = self.find_instance_by_label(self.Points, Param.label)
                cluster_level = SubPoints(Points, cluster_level_indices, **clusters[0].plot_args)
                cluster_level.add_to_plot(color=cluster_colors.get_next_color(), label=Param,
                                          s=point_size)
        # Draw Homers
        if homers:
            nearby_homer_indices = [i for i in range(len(self.active_homers)) 
                                    if self.point_in_limits(self.active_homers.points[i], limits)]
            nearby_homers = SubPoints(self.active_homers, nearby_homer_indices)
            nearby_homers.add_to_plot(color=homer_color, s=homer_size)
            # Draw Circles
            if not isinstance(circle_radii, list):
                circle_radii = [circle_radii]
            for homer_center in nearby_homers.points:
                colors = PlotColors(circle_colors)
                for radius in circle_radii:
                    plt.gca().add_artist(plt.Circle(homer_center, radius/self.nm_per_pixel, 
                                                    fill = False, color=colors.get_next_color()))
        
        # Adding box
        if box_limits is not None:
            box_x_min, box_x_max = box_limits[0]
            box_y_min, box_y_max = box_limits[1]
            width = box_x_max - box_x_min
            height = box_y_max - box_y_min
            box = plt.Rectangle((box_x_min, box_y_min), width, height, fill=False, 
                                color=box_color, linewidth=1)
            plt.gca().add_patch(box)
            

        # Plotting Scale Bar
        if scale_bar:
            plot_scale_bar(self.nm_per_pixel)
        
        # Setting axis ticks
        if not ticks:
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
        
        # Setting legend and adjusting handle sizes
        if legend:
            for handle in plt.legend(loc='upper right', fontsize=7).legend_handles:
                handle._sizes = [50]
        plt.show()

    def plot_homer(self, idx, Params=[], circle_radii=[], buffer=2000, show_points=[], dpi=150, 
                   life_act=True, cluster_centers=False, other_homers=True, scale_bar=True, 
                   ticks=True, legend=True, background_points_colors=['white', 'cyan', 'lime'],
                   cluster_colors=['red', 'orange', 'yellow', 'purple'], 
                   uniform_cluster_colors=True):
        """
        Function to plot the region around a homer center with a variety of variables to adjust
        the plot.
        
        Args:
            idx (int): Index of homer center from self.active_homers to view
            Params (list[ClusterParam], optional): List of ClusterParams to view clusters from.
            Defaults to [].
            circle_radii (list[int or float], optional): Radii (nm) to draw circles around central 
            homer. Defaults to [].
            buffer (int, optional): Radius (nm) to view around homer center (zoom level of the plot).
            Defaults to 2000 (nm).
            show_points (list[str], optional): List of labels of points in self.Points to display.
            Defaults to [].
            dpi (int, optional): DPI of image to be passed to plt.figure(dpi). Defaults to 150.
            life_act (bool, optional): Display background life act. Defaults to True.
            cluster_centers (bool, optional): Mark centers of each cluster. Defaults to False.
            other_homers (bool, optional): Show other homer centers. Defaults to True.
            scale_bar (bool, optional): Show scale bar. Defaults to True.
            ticks (bool, optional): Show plot ticks (in pixels, NOT nm). Defaults to True.
            legend (bool, optional): Show legend. Defaults to True.
            background_points_colors (list[str], optional): List of matplotlib colors to use for 
            show_points. Defaults to ['white', 'cyan', 'lime'].
            cluster_colors (list[str], optional): List of matplotlib colors to use for clusters.
            Defaults to ['red', 'orange', 'yellow', 'purple'].
            uniform_cluster_colors (bool, optional): When plotting just one param, choose whether 
            all clusters are the same color or each cluster is different. Defaults to True.
        
        Returns:
            void: Just shows the plot.
        """
        Homer = SubPoints(self.active_homers, [idx])
        homer_center = Homer.points[0]
        background_points_colors = PlotColors(background_points_colors)
        cluster_colors = PlotColors(cluster_colors)
        plt.figure(dpi=dpi)
        # Plot Life Act Background
        if life_act:
            try:
                plt.imshow(self.life_act, cmap='hot', origin='lower')
            except:
                warnings.warning("Cannot show Life_Act")
        
        # Set Plot Ranges
        buffer_px = buffer / Homer.nm_per_pixel
        plt.xlim(homer_center[0] - buffer_px, homer_center[0] + 2*buffer_px)
        plt.ylim(homer_center[1] - buffer_px, homer_center[1] + buffer_px)
        
        # Plot Background Points
        if not isinstance(show_points, list):
            show_points = [show_points]
        for i in range(len(show_points)):
            Points = self.find_instance_by_label(self.Points, show_points[i])
            # workers=-1 is for parallel processing, if running into problems, set to 1
            nearby_point_indices = KDTree(Points.points).query_ball_point(homer_center, 
                                                                          2.1*buffer_px, 
                                                                          workers=-1)
            SubPoints(Points, nearby_point_indices, s=0.75, 
                      color=background_points_colors.get_next_color()).add_to_plot()
        
        # Plot Clusters
        if not isinstance(Params, list):
            Params = [Params]
        if len(Params) == 1 and not uniform_cluster_colors:
            try:
                clusters = self.clustering_results[Params[0]]
            except:
                print(f"{Params[0]} has not been run yet, running find_clusters...")
                self.find_clusters(Params[0])
                clusters = self.clustering_results[Params[0]]
            cluster_centers = [cluster.cluster_center for cluster in clusters]
            nearby_cluster_indices = KDTree(cluster_centers).query_ball_point(homer_center, 
                                                                              2.1*buffer_px, 
                                                                              workers=-1)
            for i in nearby_cluster_indices:
                    clusters[i].add_to_plot(color=None)
        else:
            for i in range(len(Params)):
                Param = Params[i]
                try:
                    clusters = self.clustering_results[Param]
                except:
                    print(f"{Param} has not been run yet, running find_clusters...")
                    self.find_clusters(Param)
                    clusters = self.clustering_results[Param]
                cluster_centers = [cluster.cluster_center for cluster in clusters]
                nearby_cluster_indices = KDTree(cluster_centers).query_ball_point(homer_center, 
                                                                                  2.1*buffer_px, 
                                                                                  workers=-1)
                cluster_level_indices = []
                for j in nearby_cluster_indices:
                    cluster_level_indices.extend(clusters[j].indices)
                Points = self.find_instance_by_label(self.Points, Param.label)
                cluster_level = SubPoints(Points, cluster_level_indices, **clusters[0].plot_args)
                cluster_level.add_to_plot(color=cluster_colors.get_next_color(), label=Param)
        # Draw Homers
        if other_homers:
            self.active_homers.add_to_plot()
        else:
            Homer.add_to_plot()
        
        # Draw Circles
        if not isinstance(circle_radii, list):
            circle_radii = [circle_radii]
        for radius in circle_radii:
            plt.gca().add_artist(plt.Circle(homer_center, radius/Homer.nm_per_pixel, 
                                            fill = False, color='red'))
        
        
        # Plotting Scale Bar
        if scale_bar:
            plot_scale_bar(Homer.nm_per_pixel)
        
        # Setting axis ticks
        if not ticks:
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
        
        # Setting legend and adjusting handle sizes
        if legend:
            for handle in plt.legend(loc='upper right', fontsize=7).legend_handles:
                handle._sizes = [50]
        plt.show()

    def cluster_size_histogram(self, Tau_D, Params=[], bins=100, max_dark_time=500, 
                               plot_sizes_over=None, area=False, extra_FOVs=[]):
        """
        Plots of histogram of the calculated size (number of receptors) of clusters.
        
        Args:
            Tau_D (int or float): Expected dark time of one receptor
            Params (list[ClusterParam]): List of ClusterParams to plot histograms for.
            Defaults to [].
            bins (int, optional): Number of bins for the histogram. Defaults to 100.
            max_dark_time (int, optional): The maximum length of dark time (s) a cluster can have 
            to still be considered. This is to filter out false clusters. Defaults to 500 s.
            plot_sizes_over (int, optional): Will display each cluster over this size, used for 
            tuning max_dark_time. Defaults to None (plots nothing).
            area (bool, optional): Will use 2D area of clusters rather than number of receptors.
            Defaults to False.
            extra_FOVs (list[FieldOfView]), optional): A list of additional FOV frames to inlcude. Defaults to [].
        Returns:
            void: Just plots the histogram.
        """
        if not isinstance(Params, list):
            Params = [Params]
        for Param in Params:
            if area:
                cluster_sizes = self.get_cluster_areas(Tau_D, Param, max_dark_time, plot_sizes_over)
                for FOV in extra_FOVs:
                    cluster_sizes.extend(FOV.get_cluster_areas(Tau_D, Param, max_dark_time, plot_sizes_over))
            else:
                cluster_sizes = self.get_cluster_sizes(Tau_D, Param, max_dark_time, plot_sizes_over)
                for FOV in extra_FOVs:
                    cluster_sizes.extend(FOV.get_cluster_sizes(Tau_D, Param, max_dark_time, plot_sizes_over))
            self.plot_cluster_size_histogram(cluster_sizes, bins, Param, area)
    
    def cluster_size_by_distance_to_homer_center(self, Tau_D, Params=[], num_bins=20, max_dark_time=500, 
                                                 y_top=None, area=False, use_all_homers=False, 
                                                 max_distance=2000, extra_FOVs=[]):
        """
        Plots a scatter plot of cluster size vs distance to the nearest homer center. Also plots
        the average line with error bars.
        
        Args:
            Tau_D (int or float): Expected dark time of one receptor
            Params (list[ClusterParam]): List of ClusterParams to plot for. Defaults to [].
            num_bins (int, optional): Number of bins for the averaging. Defaults to 20.
            max_dark_time (int, optional): The maximum length of dark time (s) a cluster can have 
            to still be considered. This is to filter out false clusters. Defaults to 500 s.
            y_top (int, optional): Sets the upper limit on the y-axis. Defaults to None.
            area (bool, optional): Will use 2D area of clusters rather than number of receptors.
            Defaults to False.
            use_all_homers (bool, optional): Use self.all_homer_centers rather than 
            self.active_homers. Defaults to False.
            max_distance (int or float): The maximum distance from a Homer center to include.
            extra_FOVs (list[FieldOfView]), optional): A list of additional FOV frames to inlcude. Defaults to [].
        Returns:
            void: Just plots the data.
        """
        if not isinstance(Params, list):
            Params = [Params]
        for Param in Params:
            if area:
                cluster_sizes = self.get_cluster_areas(Tau_D, Param, max_dark_time)
                for FOV in extra_FOVs:
                    cluster_sizes.extend(FOV.get_cluster_areas(Tau_D, Param, max_dark_time))
            else:
                cluster_sizes = self.get_cluster_sizes(Tau_D, Param, max_dark_time)
                for FOV in extra_FOVs:
                    cluster_sizes.extend(FOV.get_cluster_sizes(Tau_D, Param, max_dark_time))
            
            clusters = [cluster for cluster in self.clustering_results[Param] 
                        if cluster.max_dark_time < max_dark_time]
            min_distances = self.get_distance_to_homer(clusters, use_all_homers)
            for FOV in extra_FOVs:
                clusters = [cluster for cluster in FOV.clustering_results[Param] 
                            if cluster.max_dark_time < max_dark_time]
                min_distances = np.concatenate((min_distances, FOV.get_distance_to_homer(clusters, use_all_homers)))
            self.plot_size_vs_distance_to_homer(cluster_sizes, min_distances, num_bins, Param, area, max_distance, y_top)
    
    def write_clusters_to_csv(self, filename, Tau_D, Params=[], max_dark_time=None, use_all_homers=False):
        """
        Writes clusters for different Params to CSV files

        This function takes as input a list of parameter sets, and for each parameter set,
        it writes the associated clusters to the CSV file. Each line in the file contains
        an index, the x and y coordinates of the cluster center (in nm), the number of subunits, 
        the area in square microns, and the distance to the nearest Homer center (in nm).

        Args:
            filename (str): The name of the CSV file to write to.
            Tau_D (float): Average dark time for a subunit
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
        for Param in Params:
            clusters = [cluster for cluster in self.clustering_results[Param]]
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

    def get_cluster_sizes(self, Tau_D, Param, max_dark_time=500, plot_sizes_over=None):
        clusters = [cluster for cluster in self.clustering_results[Param] 
                        if cluster.max_dark_time < max_dark_time]
        average_dark_times = [cluster.average_dark_time for cluster in clusters]
        cluster_sizes = [Tau_D/dark_time for dark_time in average_dark_times]                
        if plot_sizes_over is not None:
            for i in range(len(clusters)):
                if cluster_sizes[i] > plot_sizes_over:
                    print(clusters[i].max_dark_time)
                    clusters[i].plot(buffer=1500, nearby_points=True)
        return cluster_sizes
    
    def get_cluster_areas(self, Tau_D, Param, max_dark_time=500, plot_sizes_over=None):
        clusters = [cluster for cluster in self.clustering_results[Param] 
                        if cluster.max_dark_time < max_dark_time]
        cluster_areas = [cluster.cluster_area() for cluster in clusters]
        if plot_sizes_over is not None:
            for i in range(len(clusters)):
                if cluster_areas[i] > plot_sizes_over:
                    print(clusters[i].max_dark_time)
                    clusters[i].plot(buffer=1500, nearby_points=True)
        return cluster_areas
    
    def get_distance_to_homer(self, Param, max_dark_time=500, use_all_homers=False):
        clusters = [cluster for cluster in self.clustering_results[Param] 
                    if cluster.max_dark_time < max_dark_time]
        if len(clusters) == 0:
            return np.array([])
        cluster_centers = np.array([cluster.cluster_center for cluster in clusters])
        if use_all_homers:
            distances = cdist(cluster_centers, self.all_homer_centers.points, 'euclidean')
        else:
            distances = cdist(cluster_centers, self.active_homers.points, 'euclidean')
        min_distances = np.min(distances, axis=1) *self.nm_per_pixel
        return min_distances
    
    @classmethod
    def plot_size_vs_distance_to_homer(cls, cluster_sizes, min_distances, num_bins, Param, 
                                       area=False, max_distance=2000, y_top=None):
        plt.figure()
        bins = np.linspace(min_distances.min(), min_distances.max(), num=num_bins+1)
        indices = np.digitize(min_distances, bins)
        df = pd.DataFrame({'bin_index': indices, 'size': cluster_sizes})
        grouped = df.groupby('bin_index')['size'].agg(['mean', 'std'])
        grouped = grouped.reindex(range(1, num_bins + 1)).fillna(0)
        x = (bins[:-1] + bins[1:]) / 2
        plt.scatter(min_distances, cluster_sizes, s=8)
        plt.errorbar(x, grouped['mean'], yerr=grouped['std'], fmt='-o', color='orange')
        plt.title(f"Cluster Size vs. Homer Distance For: {Param}")
        plt.xlabel(f"Distance From Nearest Homer Center (nm)")
        if area:
            plt.ylabel(f"Area of {Param.label} (micron^2)")
        else:
            plt.ylabel(f"Number of {Param.label}")
        plt.ylim(bottom=0)
        plt.xlim(0, max_distance)
        if y_top is not None:
            plt.ylim(top=y_top)
        plt.show()

    @classmethod
    def plot_cluster_size_histogram(cls, cluster_sizes, bins, Param, area):
        plt.figure()
        plt.hist(cluster_sizes, bins, density=True)
        plt.title(f"Cluster Sizes for: {Param}")
        if area:
            plt.xlabel(f"Area of {Param.label} (micron^2)")
        else:
            plt.xlabel(f"Number of {Param.label}")
        plt.ylabel("Frequency")
        plt.show()