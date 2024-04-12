import numpy as np
from matplotlib import pyplot as plt
from points import BasePoints, SubPoints
from scipy.spatial import ConvexHull
from plot_helpers import plot_scale_bar

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

class Cluster(SubPoints):
    """Cluster class to handle a cluster of points from a BasePoints object.

    This class is a subclass of the SubPoints class and is used to handle a cluster of points 
    from a BasePoints object, it is also meant to be attached to a FieldOfView class 

    Attributes:
        base_points (BasePoints): The points in the cluster
        cluster_center (np.ndarray): The center point of the cluster.
        fov (FOV): The FieldOfView class the cluster is attached to.
        nearby_points (BasePoints or None): Nearby points to the cluster.
        max_dark_time (float): Maximum dark time calculated from the associated frames.
        average_dark_time (float): Average dark time calculated from the associated frames.

    Methods:
        __init__(): Initialize the Cluster class.
        __str__(): Returns a string representation of the Cluster object.
        __repr__(): Returns a developer-friendly representation of the Cluster object.
        distance_from(): Calculate the Euclidean distance from the cluster center to a given point.
        plot_life_act(): Helper function for plot() to plot 'life_act'
        plot_homers(): Helper function for plot() to plot 'homers'
        plot(): Generate a comprehensive plot with different components based on the arguments.
    """
    def __init__(self, base_points, indices, fov=None, nearby_points=None, spine=-1, **kwargs):
        """
        Initialize the Cluster class.

        Args:
            base_points (BasePoints): The BasePoints object from which the subset of points is derived.
            indices (list or np.ndarray): The indices of the points to be handled.
            fov (FieldOfView, optional): The field of view related to the cluster.
            nearby_points (BasePoints or None, optional): Nearby points related to the cluster.
            spine (int): The associated spine to the cluster. Defaults to -1 (no spine).
            **kwargs: Additional arguments for plotting.
        """
        super().__init__(base_points, indices, **kwargs)
        if len(self.points) > 0:
            self.cluster_center = self.points.mean(axis=0)
        else:
            self.cluster_center = np.array((0, 0))
        self.fov = fov
        self.nearby_points = nearby_points
        self.max_dark_time, self.average_dark_time = self.frames.get_average_dark_time(return_max=True)
        self.spine = spine
    
    def __str__(self):
        """Returns a string representation of the Cluster object."""
        return f"ClusterAt{self.cluster_center})"
    
    def __repr__(self):
        """Returns a developer-friendly representation of the Cluster object."""
        return f"ClusterAt{self.cluster_center})"
    
    def distance_from(self, point):
        """
        Calculate the Euclidean distance from the cluster center to a given point.

        Args:
            point (np.ndarray or list): A point in the same dimensional space as the cluster center.

        Returns:
            float: The Euclidean distance from the cluster center to the given point.
        """
        return np.linalg.norm(self.cluster_center - point)
    
    def cluster_area(self, nm_squared=False):
        """
        Calculate the area of the cluster by creating a Convex Hull around the points and 
        calculating its area in microns squared.
        
        Args:
            nm_squared (bool, optional): will use units of nm^2 vs micron^2. Defaults to False.

        Returns:
            float: The area of the Convex Hull of the cluster.
        """
        hull = ConvexHull(self.points)
        if nm_squared:
            return hull.area * (self.nm_per_pixel**2)
        else:
            return hull.area * (self.nm_per_pixel**2) / 100000

    def plot_life_act(self, life_act):
        """
        Helper function for plot() to plot the background 'life_act'

        Args:
            life_act (bool or array-like): The 'life_act' data to plot.
        """
        if not isinstance(life_act, bool):
            plt.imshow(life_act, cmap='hot', origin='lower')
        elif life_act:
            if self.fov is None:
                print("No FOV attached to cluster, cannot plot life_act")
            else:
                if self.fov.get_life_act() is None:
                    print("FieldOfView.life_act = None")
                    return
                plt.imshow(self.fov.get_life_act(), cmap='hot', origin='lower')

    def plot_homers(self, homers):
        """
        Helper function for plot() to plot the homer centers.

        Args:
            homers (bool or BasePoints): The 'homers' data to plot.
        """
        if not isinstance(homers, bool):
            if not isinstance(homers, BasePoints):
                homers_type = f"{type(homers)}"
                raise Exception(f"'homers' is not of class BasePoints, instead: " + homers_type)
            homers.add_to_plot()
        elif homers:
            if self.fov is None:
                print("No FOV attached to cluster, cannot plot homers")
            else:
                if self.fov.get_homers() is None:
                    print("get_homers() = None")
                    return
                self.fov.get_homers().add_to_plot()
    
    def plot(self, buffer=100, print_center=True, legend=True, scale_bar=True, time_limits=None, 
             nearby_points=False, all_points=False, homers=True, life_act=True, **kwargs):
        """
        Generate a comprehensive plot with different components based on the arguments. Will plot
        both the localizations of the cluster and its surrounding region as well as the on/off
        timeline plot.

        Args:
            buffer (int, optional): Buffer distance (nm) around the cluster. Default is 100 (nm).
            print_center (bool, optional): Whether to print the cluster center. Default is True.
            legend (bool, optional): Whether to include a legend in the plot. Default is True.
            scale_bar (bool, optional): Whether to include a scale bar in the plot. Default is True.
            time_limits (list or None, optional): Time limits for the plot in the format [min, max]. 
                Default is None.
            nearby_points (bool, optional): Whether to plot nearby points. Default is False.
            all_points (bool, optional): Whether to plot all points from the base set. 
                Default is False.
            homers (bool, optional): Whether to plot homer centers. Default is True.
            life_act (bool, optional): Whether to plot the life act background. Default is True.
            **kwargs: Additional arguments for plotting.

        """
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)

        nm_per_pixel = self.nm_per_pixel
        cluster_points = self.points
        cluster_center = self.cluster_center
        self.plot_life_act(life_act)
        if all_points:
            self.base_points.add_to_plot()
        if nearby_points:
            self.nearby_points.add_to_plot()
        self.add_to_plot(**kwargs) # Plots this cluster
        self.plot_homers(homers)
        if print_center:
            plt.scatter(cluster_center[0], cluster_center[1], marker='x', 
                        linewidth=10, s=5, color='red', label="Cluster Center")
        
        if scale_bar:
            plot_scale_bar(nm_per_pixel)
        if life_act:
            plt.title(f"{self.label} [Background: Life_act]")
        else:
            plt.title(f"{self.label}")
        if legend:
            # Increase icon sizes in legend so you can actually see the points
            for handle in plt.legend().legend_handles:
                handle._sizes = [50]
        buffer = buffer / nm_per_pixel
        plt.xlim(np.min(cluster_points[:, 0]) - buffer, np.max(cluster_points[:, 0]) + buffer)
        plt.ylim(np.min(cluster_points[:, 1]) - buffer, np.max(cluster_points[:, 1]) + buffer)
        
        # Timeline plot
        plt.subplot(1, 2, 2)
        cluster_frames = self.frames.frames
        time_per_frame = self.frames.time_per_frame
        frame_range = range(max(self.base_points.frames))
        times = [frame*time_per_frame for frame in frame_range]
        vals = [1 if frame in cluster_frames else 0 for frame in frame_range]
        plt.plot(times, vals)
        plt.title(f"Timeline for {self.label} [{len(cluster_frames)} events]")
        plt.xlabel("Time (s)")
        plt.ylabel("On [1] or Off [0]")
        plt.ylim(0, 1.3)
        if time_limits is not None:
            plt.xlim(time_limits[0], time_limits[1])
        plt.show()
