# Spherics Python module

This is a small Python module that holds some functions to be used in data
analysis in spheres.

Some of the functions that it has:

    - `spheric_distance`: calculates geodesic distance between two
        points on the sphere from their geographical longitude and latitude.
        Based on a simplification of the [Vincenty](https://en.wikipedia.org/wiki/Vincenty%27s_formulae) 
        / [Haversine](https://en.wikipedia.org/wiki/Haversine_formula) formulae.
    - `spheric_uniform`: generates a random truly uniform distribution of points
        on the sphere, outputs array of longitudes and latitudes.
    - `spheric_clusters`: generates Gaussian-like clusters of points together
        with a label. Useful for verification of clusterization algorithms.
    - Some commonplace transformations between 3D-vectors and geographical
        spherical coordinates.

## Installation

You can use `spherics.py` as is or clone the repository with

`git clone https://github.com/jesusaguado/spherics.git`

You need only `import spherics` from your python project to access all functions
within.


## License

See LICENSE file. GPLv3.
