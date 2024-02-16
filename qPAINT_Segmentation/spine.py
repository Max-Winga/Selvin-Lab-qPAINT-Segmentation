from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm

from plot_helpers import plot_scale_bar

class Spine:

    def __init__(self, label, roi, nm_per_pixel, clusters=None, homers=None):
        self.label = label
        self.roi = np.array(roi)
        self.nm_per_pixel = nm_per_pixel
        self.clusters = {} if clusters is None else clusters
        self.homers = homers

    def num_homers(self):
        if self.homers is None: return 0
        return len(self.homers)
    
    def contains_clusters(self):
        return len(self.clusters) != 0
    
    def set_homer(self, homer):
        self.homers = homer

    def set_clusters(self, Param, clusters):
        self.clusters[Param] = clusters

    def area(self, in_microns=True):
        if in_microns:
            return len(self.roi) * (self.nm_per_pixel**2) / 1000000
        return len(self.roi) * (self.nm_per_pixel**2)

    def num_clusters(self, Param):
        return len(self.clusters[Param])
    
    def plot(self, Params=None, life_act=None, plot_homers=True, Points=None, viewscale=1.5):
        if life_act is not None:
            plt.imshow(life_act, cmap='gray')
        if plot_homers:
            self.homers.add_to_plot()
        if Params is not None:
            if type(Params) is not list:
                Params = [Params]
            for Param in Params:
                num_clusters = len(self.clusters[Param])
                colors = cm.rainbow(np.linspace(0, 1, num_clusters))  # Generate colors
                
                for i, color in zip(range(num_clusters), colors):
                    self.clusters[Param][i].add_to_plot(label=f"Cluster {i}", color=color)
        if Points is not None:
            if type(Points) is not list:
                Points = [Points]
            for points in Points:
                points.add_to_plot()
        xmin, xmax = np.min(self.roi[:, 0]), np.max(self.roi[:, 0])
        ymin, ymax = np.min(self.roi[:, 1]), np.max(self.roi[:, 1])
        width, height = xmax-xmin, ymax-ymin
        plt.xlim(int(xmin + (width/2)*(1-viewscale)), int(xmax - (width/2)*(1-viewscale)))
        plt.ylim(int(ymax - (height/2)*(1-viewscale)), int(ymin + (height/2)*(1-viewscale)))
        plt.title(f"Spine {self.label}: {Params}")
        plt.legend()
        plot_scale_bar(self.nm_per_pixel)
        

