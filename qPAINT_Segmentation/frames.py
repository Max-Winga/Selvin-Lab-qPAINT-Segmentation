import numpy as np
from scipy.stats import expon
from matplotlib import pyplot as plt

class Frames():
    """
    A class used to represent store on/off times (or 'frames on') for points.

    Attributes:
        frames (numpy.ndarray): The 'on' frames of the points.
        time_per_frame (float): The time per frame.
        max_frame (int): The maximum frame number for the dataset.

    Methods:
        __len__() -> int: Returns the number of frames.
        __getitem__(idx: int) -> numpy.ndarray: Returns the frame at the specified index.
        __max__() -> int: Returns the maximum frame number.
        __min__() -> int: Returns the minimum frame number.
        get_dark_times() -> list: Calculates and returns the dark times (times between 'on' points).
        get_average_dark_time(plot: bool, return_max: bool) -> float or tuple: Calculates the 
            average dark time and optionally returns the maximum dark time.
    """
    def __init__(self, frames, time_per_frame, max_frame=None):
        """ Initializes the Frames class.

        Args:
            frames (list[int]): The list of 'on' frames.
            time_per_frame (float): The time per frame.
            max_frame (int, optional): The maximum frame number for the dataset. 
                If None, the maximum frame will be calculated from the input frames.
                Should only be None if using for BasePoints.
        """
        self.frames = np.array(frames)
        self.time_per_frame = time_per_frame
        if max_frame is None:
            self.max_frame = max(self)
        else:
            self.max_frame = max_frame
    
    def __len__(self):
        """Defines the "length" of the object as the number of frames.

        Returns:
            int: Number of frames.
        """
        return len(self.frames)
    
    def __getitem__(self, idx):
        """Defines object index access for the frames.

        Args:
            idx (int): The index of the frame to access.

        Returns:
            numpy.ndarray: The frame at the specified index.
        """
        return self.frames[idx]
    
    def __max__(self):
        """Defines the maximum frame number.

        Returns:
            int: The maximum frame number.
        """
        return max(self.frames)
    
    def __min__(self):
        """Defines the minimum frame number.

        Returns:
            int: The minimum frame number.
        """
        return min(self.frames)
    
    def get_dark_times(self):
        """Calculates the dark times (the times in between 'on' points).

        Returns:
            list: List of dark times.
        """
        times_on = [0.0]
        times_on.extend([frame*self.time_per_frame for frame in self.frames])
        times_on.extend([self.max_frame*self.time_per_frame])
        dark_times = [times_on[i] - times_on[i-1] 
                      for i in range(1, len(times_on))]
        return dark_times

    def get_average_dark_time(self, plot=False, return_max=False):
        """Calculates the average dark time.

        Args:
            plot (bool, optional): Whether to plot the dark times. Default is False.
            return_max (bool, optional): Whether to return the maximum dark time. Default is False.

        Returns:
            float or tuple: If return_max is False, returns the average dark time. 
            If return_max is True, returns a tuple of (max_dark_time, average_dark_time).
        """
        dark_times = self.get_dark_times()
        max_dark_time = max(dark_times)
        dark_times_reduced = dark_times[1:len(dark_times)-1]
        if len(dark_times_reduced) > 0:
            loc, scale = expon.fit(dark_times_reduced)
        else:
            ## Cluster consists of too few points and will crash the exponential fit
            if return_max:
                return -1, -1
            return -1

        if plot:
            x = np.linspace(expon.ppf(0.01, loc, scale),
                            expon.ppf(0.99, loc, scale), 100)
            pdf = expon.pdf(x, loc, scale)
            plt.hist(dark_times_reduced, bins=20, density=True, alpha=0.5)
            plt.plot(x, pdf, 'r-', lw=2)
            plt.show()
        if return_max:
            return max_dark_time, scale
        return scale   
