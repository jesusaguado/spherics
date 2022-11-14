# Spherics Python module

This is a very small Python module intended to recyle some common functions
to be used when dealing with geometry and data on the sphere. 

It encompasses several functions to translate between 3-dimensional position
vectors and longitude and latitude in degrees and radians, generation of
a uniform distribution of random points in the sphere given by their
longitude and latitude and generation of Gaussian-like clusters of points
on the sphere, as well as a distance function based on the Haversine
formula which can be used as a custom metric for clustering of data
on the sphere, correcting the topological error commonly encountered when
clustering longitudes and latitudes with the Euclidean metric.

## Installation

You can use `spherics.py` as is or clone the repository with

`git clone https://github.com/jesusaguado/spherics.git`

You need only `import spherics` from your python project to access all functions
within.
