class Spine:

    def __init__(self, label, roi, nm_per_pixel, clusters=None, homers=None):
        self.label = label
        self.roi = roi
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
    
