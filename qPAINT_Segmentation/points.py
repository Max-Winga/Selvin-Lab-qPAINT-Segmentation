import numpy as np
from matplotlib import pyplot as plt
from frames import Frames
import math as m

class BasePoints:
    """BasePoints class to handle points and basic plotting functionality.
    
    This class provides basic functionality to manipulate and plot points in 2D space.

    Attributes:
        label (str): Label for the class instance.
        frames (Frames): The Frames associated with these points for data that contains frames.
        nm_per_pixel (float): Scale conversion for points.
        points (np.ndarray): Points to be handled or plotted.
        plot_args (dict): Dictionary for arguments to use in plot functions.

    Methods:
        __init__(): Initialize the BasePoints class.
        __len__(): Get the number of points.
        __getitem__(): Access a specific point.
        set_plot_args(): Set the arguments to use in plot functions.
        add_to_plot(): Add the points to an existing plot.
        plot(): Plot the points.
    """
    def __init__(self, points, frames=None, nm_per_pixel=1, Tau_D=None, spine=-1, **kwargs):
        """
        Initialize the BasePoints class.
        
        Args:
            points (list or np.ndarray): List or array of points to handle or plot.
            frames (Frames): The Frames associated with these points for data that contains frames.
            nm_per_pixel (float): Scale conversion for points. Defaults to 1.
            Tau_D (float or None): Tau_D value for these points, Defaults to -1.0. 
            **kwargs: Additional arguments for plotting.
        """
        self.label = kwargs.get('label')
        self.frames = frames
        self.nm_per_pixel = nm_per_pixel
        self.points = np.array(points)
        self.Tau_D = Tau_D
        self.plot_args = kwargs

    def __len__(self):
        """Return the number of points."""
        return len(self.points)

    def __getitem__(self, idx):
        """
        Access a specific point.
        
        Args:
            idx (int): Index of the point.

        Returns:
            np.ndarray: The point at index `idx`.
        """
        return self.points[idx]

    def set_plot_args(self, **kwargs):
        """
        Set the arguments to use in plot functions.
        
        Args:
            **kwargs: Arguments for plotting.
        """
        self.plot_args = kwargs
        
    def add_to_plot(self, **kwargs):
        """
        Add the points to an existing plot.
        
        Args:
            **kwargs: Additional arguments for plotting.
        """
        args = {**self.plot_args, **kwargs}
        plt.scatter(self.points[:, 0], self.points[:, 1], **args)

    def plot(self, **kwargs):
        """
        Plot the points.
        
        Args:
            **kwargs: Additional arguments for plotting.
        """
        args = {**self.plot_args, **kwargs}
        if args.get('color') == 'white':
            args['color'] = 'black'
        plt.figure()
        self.add_to_plot(**args)
        if self.label is not None:
            plt.title(self.label)
        plt.show()

    def scale_and_floor(self, scale):
        """
        Scales the points by 'scale' and the takes the floor of each dimension to return integer coordinates
        
        Args:
            scale (float): the scale to scale the points by
        
        Returns:
            array-like: scaled coordinates
        """
        to_return = []
        for point in self.points:
            to_return.append((m.floor(point[0]*scale), m.floor(point[1]*scale)))
        return np.array(to_return)

class SubPoints(BasePoints):
    """SubPoints class to handle a subset of points from a BasePoints object.
    
    This class is a subclass of the BasePoints class and is used for handling a subset of points 
    from a BasePoints object.

    Attributes:
        indices (np.ndarray): Indices of the points to be handled from the BasePoints object.
        base_points (BasePoints): The original BasePoints object from which the subset is derived.

    Methods:
        __init__(): Initialize the SubPoints class.
        get_base_index(): Get the index of a point in the original BasePoints object.
    """
    def __init__(self, base_points, indices, **kwargs):
        """
        Initialize the SubPoints class.
        
        Args:
            base_points (BasePoints): The BasePoints object from which the subset of points is derived.
            indices (list or np.ndarray): The indices of the points to be handled.
            **kwargs: Additional arguments for plotting.
        """
        super().__init__(base_points[indices], base_points.frames, 
                         base_points.nm_per_pixel, base_points.Tau_D,**kwargs)
        self.indices = np.array(indices)
        if self.frames is not None:
            self.frames = Frames(self.frames[indices], self.frames.time_per_frame, 
                                 self.frames.max_frame)
        self.plot_args = {**base_points.plot_args, **self.plot_args}
        self.base_points = base_points

    def get_base_index(self, idx):
        """
        Get the index of a point in the original BasePoints object.
        
        Args:
            idx (int): The index of the point in the current SubPoints object.

        Returns:
            int: The index of the point in the original BasePoints object.
        """
        return self.indices[idx]
