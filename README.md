# Selvin-Lab-qPAINT-Segmentation
This project is still a work-in-progress, if you have any feedback or questions, please contact me at: mwinga2 (at) illinois.edu. 

## Library Description ##
This library was created by Max Winga for use in Gloria Lau's analysis of Glutamate receptors in single-molecule super-resolution microscopy imaging of dissociated mouse neurons using DNA paint (qPAINT) at Selvin Lab. There are two primary components to the analysis code: spine identification and cluster analysis.

Spine identification is the process of identifying dendritic spinal regions of the data using the non-super-resolution background imaged with LifeAct. To accomplish this we use two tools: DeepD3, a deep-learning based dendritic spine identification tool [1] (specifically, `DeepD3_32F.h5`); and StarDist, which uses star-convex object detection for identifying the individual spines from the binary map produced by DeepD3 [2][3]. The output of the spine identification is two files per background image which contain the starplane output and the regions-of-interest (ROIs) for each spine in CSV and JSON formats respectively to allow for processing through other software.

Field of View (FOV) analysis combines the results of the spine idenficiation with the raw super-resolution localizations for various proteins. The `FieldOfView` class is the primary interaction with this data for the user. This class takes as input localizations from Homer proteins and glutamate receptors, the LifeAct background, the identified spinal regions, and the clustering algorithms that should be run on the glutamate receptors. 

Using a pre-established DBSCAN-based methodology [4], the Homer proteins are clustered, and the centers of these clusters located. These "Homer centers" mark the approximate center point of the spine's processing and are used to approximate the distance to the center of processing for the clusters of glutamate receptors we identify. The prescence of a Homer center is one of the metrics which we use to filter out 'bad' spines from the DeepD3 spine identifications.

Once the Homer centers have been identified, we load the point localizations from their files. These points are structures in the `BasePoints`->`SubPoints`->`Cluster` class family, where `SubPoints` and `Cluster` are produced from subsets of the `BasePoints` that are loaded initially. More information about these classes can be learned through their docstrings in `points.py` and `clusters.py`. We also load the spine data from files and save it to `FieldOfView`'s `Spines` list, which wraps the spine data into a `Spine` object that will contain the points and clustering results.

With the points loaded, `FieldOfView` now moves on to the actual clustering. This process is done on a spine-by-spine basis and only considers points within the ROI of the spine under examination. Clustering can be done with any algorithm desired that can take a `BasePoints` object or descendant as input and produce a list of `Cluster` objects. These functions are provided to `FieldOfView` through a list `ClusterAlgorithm` objects, which act as a wrapper around the clustering function to allow for hashing and also to identify the points the algorithm targets (i.e. "GluA1", or "GluA2"). Once clusters are identified for a particular spine, they are saved to the `Spine` object's `clusters` dictionary, which uses the `ClusterAlgorithm` as a key to access the clusters that it found in the spine.

After identifying the clusters, the final step is to filter out the "bad" spines which do not contain both clusters and Homer centers. This step can be disabled through the `filter_spines` argument to the `FieldOfView` initalization.

Finally, with analysis complete, the results of the clustering can be printed to CSV using the `write_clusters_to_csv` method of `FieldOfView`. This will save the cluster sizes (in number of subunits), the area of the clusters, and the cluster's distance to Homer for each spine. The number of subunits is calculated via the dynamics of the blinking of the dyes used for the super-resolution microscopy. The dyes blink on and off and do so with an average "dark time". Therefore, when looking at a large group of localizations, we can divide the average dark time of a single subunit by the (smaller) average dark time of the whole cluster, thus providing an estimate for the number of subunits. Examining the "trace" of the cluster (the timeline of when single "hits" occurs) can give insight into the validity of the cluster. Clusters with large gaps in the trace may not be true signal and thus can be discarded by filtering using the `max_dark_time` variable. 

**IMPORTANT NOTE FOR M1/M2 MACBOOK USERS:**
Multithreading is enabled by default to boost performance, you will need to disable this by setting `multithreading=1` when you initialize the `FieldOfView` class.

## Environment Setup ##
After downloading this repository, you will need to set up the anaconda python environments, I reccommend working through Visual Studio Code. This library uses two different anaconda environments to deal with conflicts between tensorflow and everything else. 'fov_analysis' should be used for the actual analysis of data and 'spine_identification' is only for use when generating spinemaps with tensorflow  

To recreate the Anaconda environments for spine identifcation and FOV analysis, run the following commands in the command line while in the main folder for the repository.

```bash
conda env create -f env\fov_analysis.yml
conda env create -f env\spine_identification.yml
```

To start either anaconda environment run:
```bash
conda activate fov_analysis
conda activate spine_identification
```

Example usage is included in the jupyter notebook, note that example data is not included and should be placed in the "Examples" folder. If the filenames are different the paths will need to be changed in the notebook.

## Citations ##
[1] Fernholz MHP, Guggiana Nilo DA, Bonhoeffer T, Kist AM (2024) DeepD3, an open framework for automated quantification of dendritic spines. PLoS Comput Biol 20(2): e1011774. https://doi.org/10.1371/journal.pcbi.1011774

[2] Schmidt, U., Weigert, M., Broaddus, C., & Myers, G. (2018). Cell Detection with Star-Convex Polygons. Medical Image Computing and Computer Assisted Intervention - MICCAI 2018 - 21st International Conference, Granada, Spain, September 16-20, 2018, Proceedings, Part II, 265–273. doi:10.1007/978-3-030-00934-2_30 

[3] Weigert, M., Schmidt, U., Haase, R., Sugawara, K., & Myers, G. (2020, March). Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy. The IEEE Winter Conference on Applications of Computer Vision (WACV). doi:10.1109/WACV45572.2020.9093435 

[4] Youn, Y., Lau, G. W., Lee, Y., Maity, B. K., Gouaux, E., Chung, H. J., & Selvin, P. R. (2023). Quantitative DNA-PAINT imaging of AMPA receptors in live neurons. Cell Reports Methods, 3(2), 100408. doi:10.1016/j.crmeth.2023.100408
