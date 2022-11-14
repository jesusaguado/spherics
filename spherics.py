import numpy as np
import math

def longlat_to_vector(long, lat, R = 1, angle = 'deg'):
    """
    This function takes a longitude and a latitude and gives the corresponding
    position vector in Euclidean 3-dimensional space on the sphere.
    Default radius is set to 1.

    The output is a 3-tuple. Input angles can be given in 'deg' or 'rad',
    by the value of the optional parameter `angle`.
    """

    if angle == 'deg':
        long = math.radians(long)
        lat = math.radians(lat)
    elif angle != 'radian':
        raise Exception("Angle parameter must be either 'deg' or 'rad'.")

    x = np.cos(long) * np.cos(lat)
    y = np.sin(long) * np.cos(lat)
    z = np.sin(lat)

    return x, y, z

def vector_to_longlat( x, y, z, angle = 'deg'):
    """
    This function computes the longitude and latitude of the input 3-dimensional
    point x, y, z, which is not assumed to be normalized before-hand.

    If it is the zero vector it will raise an Exception. Output longitude
    and latitude can be in either 'deg' or 'rad' for degrees and radians.
    """

    if angle not in ['deg', 'rad']:
        raise Exception("Angle parameter must be either 'deg' or 'rad'.")

    #r = math.sqrt(x**2 + y**2 + z**2)
    #x, y, z = x / r, y / r, z / r

    long = math.atan2(y, x)
    lat = math.asin(z)

    if angle == 'deg':
        long = math.degrees(long)
        lat = math.degrees(lat)
    return long, lat

def spheric_distance(p, q, radius = 'unit', units = None, angle = 'deg'):
    """
    This function calculates the distance between two points on the
    sphere from their longitude and longitude. Some optional arguments
    are radius ('unit', 'earth') and units ('m', 'km').
    """

    if angle not in ['deg', 'rad']:
        raise Exception("Angle parameter must be either 'deg' or 'rad'.")
    if radius not in ['unit', 'earth']:
        raise Exception("Radius parameter must be either 'unit' or 'earth'.")
    if units not in ['m', 'km', None]:
        raise Exception("Units parameter must be either 'm', 'km' or None.")

    long1, lat1 = p
    long2, lat2 = q
    if angle == 'deg':
        long1 = math.radians(long1)
        long2 = math.radians(long2)
        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)

    delta_long = long1 - long2
    
    up =  math.sqrt((np.cos(lat2) * np.sin(delta_long))**2+\
            (np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2)*\
            np.cos(delta_long))**2)
    down = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) *\
            np.cos(delta_long)

    sigma = math.atan2(up, down)

    if radius == 'unit':
        R = 1
    elif radius == 'earth':
        R = 6371000 # Earth radius in meters

    return R * sigma


def spheric_uniform(N, angle = 'deg'):
    """
    This function generates random points on the sphere by normalizing
    the radius of the 3-dimensional gaussian distribution.
    It outputs a NumPy array X of shape (N,2) with the longitudes and
    latitudes of the points on the sphere, in degrees or radians.
    """
    Z = np.random.multivariate_normal(np.array([0., 0., 0.]), \
            np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), N)

    norms = np.linalg.norm(Z, axis = 1)

    Z = Z / norms.reshape(N,1)

    X = np.zeros((N, 2))
    for i in range(0,N):
        X[i, 0], X[i, 1] = vector_to_longlat(Z[i, 0], Z[i, 1], Z[i, 2], angle = angle)
    return X


def sphere_cluster(centroids, N, std = 0.1, angle = 'deg'):
    """
    This function takes some non-zero centroids in latitude and longitude
    and generates a small Gaussian-like cluster of nearby points. It
    also returns the label of the data point, useful for clustering
    verification.

    Inputs: centroids: either an array of shape (m, 2) with the longitudes
                        and latitudes of the centroids, or a list of m lists.

            N: the number of data points to generate per centroid

            std: the standard deviation of the point

    Outputs:

        X: a NumPy array of shape (N,2) with the longitudes and latitudes
            of the generated data points.
        y: a NumPy array of shape (N) with the integer label in 0,... len(centroids)
    """

    if angle not in ['deg', 'rad']:
        raise Exception("Angle parameter must be either 'deg' or 'rad'.")

    centroids = np.array(centroids)

    l = len(centroids)

    sigma = std**2

    vector_centroids = np.zeros((l, 3))

    X = np.zeros((1,2))
    #X = np.empty((N*l, 2))
    print('Initial X: ', X)
    y = np.zeros(1)
    #y = np.empty((N*l))
    print('Initial y: ', y)

    for i in range(0, l):
        long = centroids[i, 0]
        lat = centroids[i, 1]
        vector_centroids[i, 0], vector_centroids[i,1], vector_centroids[i,2] = longlat_to_vector(long, lat, angle = angle)

    for i in range(0, l):
        #long = centroids[i, 0]
        #lat = centroids[i, 1]
        #centroid = np.array(longlat_to_vect(long, lat, angle = angle))
        Z = np.random.multivariate_normal(vector_centroids[i],  \
                np.array([[sigma, 0., 0.], [0., sigma, 0.], [0., 0., sigma]]), N)
        norms = np.linalg.norm(Z, axis = 1)

        Z = Z / norms.reshape(N,1)

        Y = np.zeros((N, 2))
        temp_labels = np.repeat(i, N)
        print('LABEL: ', temp_labels)

        for i in range(0,N):
            Y[i, 0], Y[i, 1] = vector_to_longlat(Z[i, 0], Z[i, 1], Z[i, 2], angle = angle)
        print('Iteration: ', i)
        print('Generated point cluster:')
        print(Y)

        X = np.concatenate((X, Y))
        y = np.concatenate((y, temp_labels))

    X = X[1:]
    y = y[1:].astype(int)

    return X, y

