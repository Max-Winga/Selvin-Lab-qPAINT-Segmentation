import numpy as np
from matplotlib import pyplot as plt
from points import BasePoints, SubPoints
from scipy.spatial import ConvexHull
from plot_helpers import plot_scale_bar

class ClusterParam():
    """
    A class used to store clustering parameters (eps and min_samples) for DBSCAN and label 
    for the points to cluster.

    Attributes:
        pps (float): nm/pixel 
        min_samples (int): The number of samples in a neighborhood for a point to be considered as 
            a core point.
        params (tuple): A tuple containing eps and min_samples parameters.
        density (float): The density of the cluster, calculated as min_samples divided by the area 
            of the cluster.
        label (str): Label for the points to cluster.

    Methods:
        __eq__(other: ClusterParam) -> bool: Checks if the current instance is equal to the 
            'other' instance.
        __lt__(other: ClusterParam) -> bool: Checks if the current instance's density is less 
            than the 'other' instance's.
        __gt__(other: ClusterParam) -> bool: Checks if the current instance's density is greater 
            than the 'other' instance's.
        __le__(other: ClusterParam) -> bool: Checks if the current instance's density is less than 
            or equal to the 'other' instance's.
        __ge__(other: ClusterParam) -> bool: Checks if the current instance's density is greater 
            than or equal to the 'other' instance's.
        __str__() -> str: Returns a string representation of the instance.
        __repr__() -> str: Returns a string representation of the instance suitable for development.
        __hash__() -> int: Returns the hash value of the instance.
        __getitem__(idx: int) -> float or int: Returns the eps or min_samples value based on the index.
    """
    def __init__(self, eps, min_samples, label=""):
        """
        Initializes the ClusterParam class.

        Args:
            eps (float): The maximum distance between two samples for one to be considered 
                as in the neighborhood of the other. This is not a maximum bound on the 
                distances of points within a cluster.
            min_samples (int): The number of samples in a neighborhood for a point to be 
                considered as a core point.
            label (str, optional): Label for the points to cluster. Default is an empty string.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.params = (eps, min_samples)
        self.density = min_samples / (np.pi * (eps ** 2))
        self.label=label

    def __eq__(self, other):
        """ 
        Checks if the current instance is equal to the 'other' instance.

        Args:
            other (ClusterParam): The other instance of ClusterParam.

        Returns:
            bool: True if the current instance is equal to the 'other' instance, False otherwise.
        """
        if not isinstance(other, ClusterParam):
            return NotImplemented
        return (self.eps == other.eps and 
                self.min_samples == other.min_samples and 
                self.label == other.label)
    
    def __lt__(self, other):
        """ 
        Checks if the current instance's density is less than the 'other' instance's.

        Args:
            other (ClusterParam): The other instance of ClusterParam.

        Returns:
            bool: True if the current instance's density is less than the 'other' instance's, 
                  False otherwise.
        """
        return self.density < other.density
    
    def __gt__(self, other):
        """ 
        Checks if the current instance's density is greater than the 'other' instance's.

        Args:
            other (ClusterParam): The other instance of ClusterParam.

        Returns:
            bool: True if the current instance's density is greater than the 'other' instance's, 
                  False otherwise.
        """ 
        return self.density > other.density
    
    def __le__(self, other):
        """ 
        Checks if the current instance's density is less than or equal to the 'other' instance's.

        Args:
            other (ClusterParam): The other instance of ClusterParam.

        Returns:
            bool: True if the current instance's density is less than or equal to the 'other' 
                  instance's, False otherwise.
        """
        return self.density <= other.density
    
    def __ge__(self, other):
        """ 
        Checks if the current instance's density is greater than or equal to the 'other' instance's.

        Args:
            other (ClusterParam): The other instance of ClusterParam.

        Returns:
            bool: True if the current instance's density is greater than or equal to the 'other' 
                  instance's, False otherwise.
        """
        return self.density >= other.density
    
    def __str__(self):
        """ 
        Returns a string representation of the instance.

        Returns:
            str: A string representation of the instance in the format 
                 "label(eps=eps, min_samples=min_samples)".
        """
        return f"{self.label}(eps={self.eps}, min_samples={self.min_samples})"
    
    def __repr__(self):
        """ 
        Returns a string representation of the instance suitable for development.

        Returns:
            str: A string representation of the instance in the format 
                 "label(eps=eps, min_samples=min_samples)".
        """ 
        return f"{self.label}(eps={self.eps}, min_samples={self.min_samples})"
    
    def __hash__(self):
        """ 
        Returns the hash value of the instance.

        Returns:
            int: The hash value of the instance, computed as the hash of a tuple containing 
                 eps, min_samples, and label.
        """
        return hash((self.eps, self.min_samples, self.label))
    
    def __getitem__(self, idx):
        """ 
        Returns the eps or min_samples value based on the index.

        Args:
            idx (int): The index (0 or 1) to access eps or min_samples, respectively.

        Returns:
            float or int: The eps or min_samples value based on the index.

        Raises:
            IndexError: If an index other than 0 or 1 is provided.
        """
        if idx == 0:
            return self.eps
        elif idx == 1:
            return self.min_samples
        else:
            raise IndexError("""Only indices 0 (eps) and 1 (min_samples)
                              are valid for ClusterParam""")

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
        self.cluster_center = base_points[indices[0]] ### CHANGED FOR NEW METHOD
        self.fov = fov
        self.nearby_points = nearby_points
        dark_times = self.frames.get_average_dark_time(return_max=True)
        self.max_dark_time, self.average_dark_time = dark_times
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
                if self.fov.life_act is None:
                    print("FieldOfView.life_act = None")
                    return
                plt.imshow(self.fov.life_act, cmap='hot', origin='lower')

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
                if self.fov.all_homer_centers is None:
                    print("FieldOfView.all_homer_centers = None")
                    return
                self.fov.all_homer_centers.add_to_plot()
    
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
        plt.tight_layout()
        plt.show()
