from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm

from plot_helpers import plot_scale_bar

class Spine:

    def __init__(self, label, roi, nm_per_pixel, clusters=None, homers=None, points=None):
        self.label = label
        self.roi = np.array(roi)
        self.nm_per_pixel = nm_per_pixel
        self.clusters = {} if clusters is None else clusters
        self.homers = homers
        self.points = {} if points is None else points

    def num_homers(self):
        if self.homers is None: return 0
        return len(self.homers)
    
    def contains_clusters(self, alg=None):
        if alg:
            if len(self.get_clusters(alg)) > 0:
                return True
            return False
        for alg in self.clusters:
            if len(self.get_clusters(alg)) > 0:
                return True
        return False
    
    def set_homer(self, homer):
        self.homers = homer

    def set_clusters(self, alg, clusters):
        self.clusters[alg] = clusters

    def get_clusters(self, alg=None):
        if alg:
            if alg in self.clusters.keys():
                return self.clusters[alg]
            return []
        return self.clusters

    def area(self, in_microns=True):
        if in_microns:
            return len(self.roi) * (self.nm_per_pixel**2) / 1000000
        return len(self.roi) * (self.nm_per_pixel**2)

    def num_clusters(self, alg):
        return len(self.get_clusters(alg))
    
    def plot(self, algs=None, life_act=None, plot_homers=True, Points=None, viewscale=1.5):
        plt.figure()
        if life_act is not None:
            plt.imshow(life_act, cmap='magma')
        if plot_homers:
            self.homers.add_to_plot()
        if Points is not None:
            if type(Points) is not list:
                Points = [Points]
            for points in Points:
                points.add_to_plot(color='white')
        if algs is not None:
            if type(algs) is not list:
                algs = [algs]
            for alg in algs:
                clusters = self.get_clusters(alg)
                num_clusters = len(clusters)
                colors = cm.rainbow(np.linspace(0, 1, num_clusters))  # Generate colors
                for i, color in zip(range(num_clusters), colors):
                    clusters[i].add_to_plot(label=f"Cluster {i}", color=color)
        
        xmin, xmax = np.min(self.roi[:, 0]), np.max(self.roi[:, 0])
        ymin, ymax = np.min(self.roi[:, 1]), np.max(self.roi[:, 1])
        width, height = xmax-xmin, ymax-ymin
        plt.xlim(int(xmin + (width/2)*(1-viewscale)), int(xmax - (width/2)*(1-viewscale)))
        plt.ylim(int(ymax - (height/2)*(1-viewscale)), int(ymin + (height/2)*(1-viewscale)))
        plt.title(f"Spine {self.label}: ClusterAlg {alg}")
        plt.legend()
        plot_scale_bar(self.nm_per_pixel)
        plt.show()
        

