import numpy as np
from matplotlib import pyplot as plt
from points import BasePoints, SubPoints
from scipy.spatial import ConvexHull
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import csv

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
        self.nearby_points = nearby_points
        self.max_dark_time, self.average_dark_time = self.frames.get_average_dark_time(return_max=True)
        if type(base_points) == Cluster:
            self.spine = base_points.spine
            self.fov = base_points.fov
        else:
            self.spine = spine
            self.fov = fov
        if self.fov is not None and self.spine is not None and self.fov.Spines[self.spine].homers is not None:
            try:
                self.distance_to_nearest_homer = min([self.distance_from(homer) for homer in self.fov.Spines[self.spine].homers]) * self.nm_per_pixel
            except:
                print(self.fov)
                print(self.spine)
                print(self.fov.Spines[self.spine].homers)
        else:
            self.distance_to_nearest_homer = None
        self.cluster_number = 0
        
        # Calculate the number of subunits based on the average dark time
        if self.average_dark_time > 0:
            self.subunits = self.Tau_D / self.average_dark_time
        else:
            self.subunits = 0
    
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
            return hull.area * (self.nm_per_pixel**2) / 1000000

    def MSE_loss(self):
        # Calculate the MSE loss
        if len(self.points) == 0:
            self.MSE_frames_loss = 1
        else:
            self.MSE_frames_loss = self.calculate_MSE_frame_loss()
        return self.MSE_frames_loss

    def calculate_MSE_frame_loss(self, return_arrays=False):
        frames = np.zeros(self.frames.max_frame)
        for frame in self.frames.frames:
            frames[frame] = 1
        cdf_array = np.zeros_like(frames)
        linear_array = np.zeros_like(frames)
        current_cdf_val = 0
        current_linear_val = 0
        for i in range(len(frames)):
            linear_array[i] = current_linear_val
            current_linear_val += 1
            if (frames[i]):
                current_cdf_val += 1
            cdf_array[i] = current_cdf_val
        cdf_array = cdf_array / current_cdf_val
        linear_array = linear_array / current_linear_val
        avg_MSE_loss = sum([(linear_array[i]-cdf_array[i])**2 for i in range(len(cdf_array))]) / len(cdf_array)
        if return_arrays:
            return cdf_array, linear_array, avg_MSE_loss
        return avg_MSE_loss
    
    def plot(self, buffer=300):
        fig, (visual, timeline) = plt.subplots(1, 2, dpi=250, figsize=(9, 4.5))
        fig.suptitle(f"Visualization of Cluster {self.cluster_number} with {self.subunits} Subunits from Spine {self.spine}")

        buffer_px = buffer // self.nm_per_pixel

        spine = self.fov.Spines[self.spine]
        roi = spine.roi
        roi_x = roi[:, 0]
        roi_y = roi[:, 1]

        life_act = self.fov.life_act

        visual.set_title(f"Spine {self.spine} and Cluster {self.cluster_number} [Background: LifeAct]")

        # Display the life_act image
        visual.imshow(life_act, cmap='gray')

        # Overlay the spine mask on top of the life_act image
        mask = np.zeros_like(life_act, dtype=bool)
        mask[roi_y, roi_x] = True

        color_mask = np.zeros((life_act.shape[0], life_act.shape[1], 4))
        color_mask[mask] = [1, 0, 0, 0.1]  # Red color with alpha=0.1
        visual.imshow(color_mask, label="Spine ROI")

        # Plot Points
        all_points = spine.points[self.label]
        visual.scatter(all_points[:, 0], all_points[:, 1], label=f"All {self.label}", s=0.1, c='white')
        visual.scatter(self.points[:, 0], self.points[:, 1], label=f"Cluster {self.cluster_number}", s=0.1, c='blue')
        visual.scatter(spine.homers[:, 0], spine.homers[:, 1], marker='v', color='chartreuse', s=100, edgecolor='black', label="Homer Center")

        visual.set_xlim(np.min(roi_x) - buffer_px, np.max(roi_x) + buffer_px)
        visual.set_ylim(np.max(roi_y) + buffer_px, np.min(roi_y) - buffer_px)
        visual.legend(fontsize='x-small')

        fontprops = fm.FontProperties(size=8)
        scalebar = AnchoredSizeBar(visual.transData,
                                1000/self.nm_per_pixel,  # length of scale bar
                                '1 micron',  # label
                                'lower right',  # position
                                pad=0.1,
                                color='white',
                                frameon=False,
                                size_vertical=0.05,
                                fontproperties=fontprops)
        visual.add_artist(scalebar)

        timeline.set_title(f"Event Timeline: {len(self.frames)} Events")

        frames = np.zeros(self.frames.max_frame)
        for frame in self.frames.frames:
            frames[frame] = 1

        # Calculate the time values based on the frame number and frames per second
        time_values = np.arange(len(frames)) * self.frames.time_per_frame

        timeline.plot(time_values, frames)
        timeline.set_yticks([0, 1])  # Set y-ticks to 0 and 1
        timeline.set_yticklabels(['Off', 'On'])  # Set y-tick labels to 'Off' and 'On'
        timeline.set_xlabel("Time (seconds)")  # Set the x-axis label to "Time (seconds)"

        plt.tight_layout()
        plt.show()

    def plot_frames_cdf(self, buffer=300):
        fig, (visual, timeline, cdf) = plt.subplots(1, 3, dpi=250, figsize=(13.5, 4.5))
        fig.suptitle(f"Frames CDF of Cluster {self.cluster_number} with {self.subunits} Subunits from Spine {self.spine}")

        buffer_px = buffer // self.nm_per_pixel

        spine = self.fov.Spines[self.spine]
        roi = spine.roi
        roi_x = roi[:, 0]
        roi_y = roi[:, 1]

        life_act = self.fov.life_act

        visual.set_title(f"Spine {self.spine} and Cluster {self.cluster_number} [Background: LifeAct]")

        # Display the life_act image
        visual.imshow(life_act, cmap='gray')

        # Overlay the spine mask on top of the life_act image
        mask = np.zeros_like(life_act, dtype=bool)
        mask[roi_y, roi_x] = True

        color_mask = np.zeros((life_act.shape[0], life_act.shape[1], 4))
        color_mask[mask] = [1, 0, 0, 0.1]  # Red color with alpha=0.1
        visual.imshow(color_mask, label="Spine ROI")

        # Plot Points
        all_points = spine.points[self.label]
        visual.scatter(all_points[:, 0], all_points[:, 1], label=f"All {self.label}", s=0.1, c='white')
        visual.scatter(self.points[:, 0], self.points[:, 1], label=f"Cluster {self.cluster_number}", s=0.1, c='blue')
        visual.scatter(spine.homers[:, 0], spine.homers[:, 1], marker='v', color='chartreuse', s=100, edgecolor='black', label="Homer Center")

        visual.set_xlim(np.min(roi_x) - buffer_px, np.max(roi_x) + buffer_px)
        visual.set_ylim(np.max(roi_y) + buffer_px, np.min(roi_y) - buffer_px)
        visual.legend(fontsize='x-small')

        fontprops = fm.FontProperties(size=8)
        scalebar = AnchoredSizeBar(visual.transData,
                                1000/self.nm_per_pixel,  # length of scale bar
                                '1 micron',  # label
                                'lower right',  # position
                                pad=0.1,
                                color='white',
                                frameon=False,
                                size_vertical=0.05,
                                fontproperties=fontprops)
        visual.add_artist(scalebar)

        timeline.set_title(f"Event Timeline: {len(self.frames)} Events")

        frames = np.zeros(self.frames.max_frame)
        for frame in self.frames.frames:
            frames[frame] = 1

        # Calculate the time values based on the frame number and frames per second
        time_values = np.arange(len(frames)) * self.frames.time_per_frame

        timeline.plot(time_values, frames)
        timeline.set_yticks([0, 1])  # Set y-ticks to 0 and 1
        timeline.set_yticklabels(['Off', 'On'])  # Set y-tick labels to 'Off' and 'On'
        timeline.set_xlabel("Time (seconds)")  # Set the x-axis label to "Time (seconds)"

        cdf_array, linear_array, avg_MSE_loss = self.calculate_MSE_frame_loss(return_arrays=True)
        cdf.plot(time_values, cdf_array)
        cdf.plot(time_values, linear_array)
        cdf.set_yticks([0, 1])
        cdf.set_ylabel("CDF")
        cdf.set_xlabel("Time (seconds)")
        cdf.set_title(f"CDF of Frames: Average MSE Loss = {avg_MSE_loss}")
        plt.tight_layout()
        plt.show()
            
    def write_cluster_points_to_csv(self, filename, include_distance_to_homer=True):
        if include_distance_to_homer:
            include_distance_to_homer = self.fov is not None and self.spine is not None and self.fov.Spines[self.spine].homers is not None
        
        if include_distance_to_homer:
            lines = [['Point Index', 'x (nm)', 'y (nm)', 'Frame', 'Distance to Nearest Homer Center (nm)']]
        else:
            lines = [['Point Index', 'x (nm)', 'y (nm)', 'Frame']]
        
        for point_idx in range(len(self.points)):
            point = self.points[point_idx]
            x = point[0] * self.nm_per_pixel
            y = point[1] * self.nm_per_pixel
            frame = self.frames[point_idx]
            if include_distance_to_homer:
                distance_to_homer = min([np.linalg.norm(point - homer) for homer in self.fov.Spines[self.spine].homers]) * self.nm_per_pixel
                lines.append([point_idx, x, y, frame, distance_to_homer])
            else:
                lines.append([point_idx, x, y, frame])
        
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(lines)
        
        print(f"{filename} created successfully!")