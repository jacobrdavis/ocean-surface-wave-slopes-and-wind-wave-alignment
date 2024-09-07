"""
Basic geographic calculations on the spherical earth.

A summary of the basic equations can be found here:

    http://www.movable-type.co.uk/scripts/latlong.html

For proper geodesic algorithms and implementations, see:

Karney, C.F.F. Algorithms for geodesics. J Geod 87, 43-55 (2013).
    https://doi.org/10.1007/s00190-012-0578-z

GeographicLib - https://geographiclib.sourceforge.io/index.html

"""


from typing import Tuple, Union

import numpy as np
import pyproj
from pyproj import Transformer


def euclidean_distance(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray
) -> np.ndarray:
    """ Compute the euclidean distance between two points. """
    dx = x2 - x1
    dy = y2 - y1
    return np.sqrt(dx**2 + dy**2)


def great_circle_pathwise(
    longitude: np.ndarray,
    latitude: np.ndarray,
    earth_radius: float = 6378.137,
    mod_bearing: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the great circle distance (km) and true fore bearing (deg) along a
    path using adjacent values in `longitude` and `latitude`.

    For two longitude and latitude pairs, the great circle distance is the
    shortest distance between the two points along the Earth's surface. This
    distance is calculated using the Haversine formula. The first instance in
    longitude and latitude is designated as point `a`; the second instance is
    point `b`. The true fore bearing is the bearing, measured from true north,
    of `b` as seen from `a`.

    Note:
        When given `latitude` and `longitude` of shape (n,), n > 1, the great
        circle distance and fore bearing will be calculated between adjacent
        entries such that the returned arrays will be of shape (n-1,). To
        compute the great circle distance and bearings for distinct pairs of
        coordinates, use `great_circle_pairwise`.

    Args:
        longitude (np.array): of shape (n,) in units of decimal degrees
        latitude (np.array): of shape (n,) in units of decimal degrees
        earth_radius (float, optional): earth's radius in units of km.
            Defaults to 6378.137 km (WGS-84).
        mod_bearing (bool, optional): return bearings modulo 360 deg.
            Defaults to True.

    Raises:
        ValueError: if longitude or latitude are less than size of 2.

    Returns:
        Tuple[np.array, np.array]: great circle distances (in km) and true fore
        bearings between adjacent longitude and latitude pairs; shape (n-1,)

    Example:
        A trajectory along the Earth's equator::
    ```
    >> longitude = np.array([0, 1, 2, 3])
    >> latitude = np.array([0, 0, 0, 0])
    >> distance_km, bearing_deg = haversine_distance(longitude, latitude)
    >> distance_km
        array([111.19, 111.15, 111.08])  # 111 km ~ 60 nm
    >> bearing_deg
        array([90., 90., 90.]))
    ```
    """
    longitude = np.asarray(longitude)
    latitude = np.asarray(latitude)

    if longitude.size <= 1 or latitude.size <= 1:
        raise ValueError("`longitude` and `latitude` must have size"
                         " of at least 2.")

    # Offset the longitude and latitude by one index to compute the haversine
    # distance and bearing between adjacent positions.
    longitude_a = longitude[0:-1]
    longitude_b = longitude[1:]
    latitude_a = latitude[0:-1]
    latitude_b = latitude[1:]

    # Pass pairs the core pairwise function.
    distance_km, bearing_deg = great_circle_pairwise(longitude_a, latitude_a,
                                                     longitude_b, latitude_b,
                                                     earth_radius=earth_radius,
                                                     mod_bearing=mod_bearing)
    return distance_km, bearing_deg


def great_circle_pairwise(
    longitude_a: np.ndarray,
    latitude_a: np.ndarray,
    longitude_b: np.ndarray,
    latitude_b: np.ndarray,
    earth_radius: float = 6378.137,
    mod_bearing: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the great circle distance (km) and true fore bearing (deg) between
    pairs of observations in input arrays `longitude_a` and `longitude_b` and
    `latitude_a` and `latitude_b`.

    For two longitude and latitude pairs, the great circle distance is the
    shortest distance between the two points along the Earth's surface. This
    distance is calculated using the Haversine formula. The instances in
    `longitude_a` and `latitude_a` are designated as point `a`; the instances
    in `longitude_b` and `latitude_b` then form point `b`. The true fore
    bearing is the bearing, measured from true north, of `b` as seen from `a`.

    Note:
        When given `latitude_a/b` and `longitude_a/b` of shape (n,), n > 1,
        the great circle distance and fore bearing will be calculated between
        `a` and `b` entries such that the returned arrays will be of shape
        (n,). To compute the great circle distance and bearings between
        adjacent coordinates of single longitude and latitude arrays (i.e.,
        along a trajectory), use `great_circle_pathwise`.

    Args:
        longitude_a (np.array): of shape (n,) in units of decimal degrees
        latitude (np.array): of shape (n,) in units of decimal degrees
        earth_radius (float, optional): earth's radius in units of km. Defaults to 6378.137 km (WGS-84)
        mod_bearing (bool, optional): return bearings modulo 360 deg. Defaults to True.

    Returns:
        Tuple[np.array, np.array]: great circle distances (in km) and true fore
        bearings between adjacent longitude and latitude pairs; shape (n,)
    """
    # Convert decimal degrees to radians
    longitude_a_rad, latitude_a_rad = map(np.radians, [longitude_a, latitude_a])
    longitude_b_rad, latitude_b_rad = map(np.radians, [longitude_b, latitude_b])

    # Difference longitude and latitude
    longitude_difference = longitude_b_rad - longitude_a_rad
    latitude_difference = latitude_b_rad - latitude_a_rad

    # Haversine formula
    a_1 = np.sin(latitude_difference / 2) ** 2
    a_2 = np.cos(latitude_a_rad)
    a_3 = np.cos(latitude_b_rad)
    a_4 = np.sin(longitude_difference / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a_1 + a_2 * a_3 * a_4))
    distance_km = earth_radius * c

    # True bearing
    bearing_num = np.cos(latitude_b_rad) * np.sin(-longitude_difference)
    bearing_den_1 = np.cos(latitude_a_rad) * np.sin(latitude_b_rad)
    bearing_den_2 = - np.sin(latitude_a_rad) * np.cos(latitude_b_rad) * np.cos(longitude_difference)
    bearing_deg = -np.degrees(np.arctan2(bearing_num, bearing_den_1 + bearing_den_2))

    if mod_bearing:
        bearing_deg = bearing_deg % 360

    return distance_km, bearing_deg


def destination_coordinates(
    origin_longitude: np.ndarray,
    origin_latitude: np.ndarray,
    distance: float,
    bearing: float,
    earth_radius: float = 6378.137
) -> Tuple[np.ndarray, np.ndarray]:
    """ Return the destination given an origin, distance, and bearing.

    Computes the destination longitude and latitude based on the distance and
    bearing from an origin longitude and latitude.

    Note:
        This is the inverse operation of `great_circle_pairwise` which computes
        the distance and bearing between two points `a` and `b` (i.e., the
        origin and destination).

    Args:
        origin_longitude (np.ndarray): in decimal degrees with shape (n,)
        origin_latitude (np.ndarray): in decimal degrees with shape (n,)
        distance (float): great circle distance with the same units as
            `earth_radius` (the default `earth_radius` is in km).
        bearing (float): True bearing from the origin to the destination
            in degrees.
        earth_radius (float, optional): Earth's radius in units of km.
            Defaults to 6378.137 km (WGS-84)

    Returns:
        Tuple[np.ndarray, np.ndarray]: longitude and latitude of the
            destination each with shape (n,).
    """
    # Convert decimal degrees to radians
    origin_longitude_rad = np.radians(origin_longitude)
    origin_latitude_rad = np.radians(origin_latitude)

    bearing_rad = np.radians(bearing)
    angular_distance = distance/earth_radius

    # Calculate the latitude at the destination
    lat_term_1 = np.sin(origin_latitude_rad) * np.cos(angular_distance)
    lat_term_2 = np.cos(origin_latitude_rad) * np.sin(angular_distance) * np.cos(bearing_rad)
    destination_latitude_rad = np.arcsin(lat_term_1 + lat_term_2)

    # Calculate the longitude at the destination
    lon_term_1 = np.sin(bearing_rad) * np.sin(angular_distance) * np.cos(origin_latitude_rad)
    lon_term_2 = np.cos(angular_distance)
    lon_term_3 = np.sin(origin_latitude_rad) * np.sin(destination_latitude_rad)
    lon_arctan = np.arctan2(lon_term_1, lon_term_2 + lon_term_3)
    destination_longitude_rad = origin_longitude_rad + lon_arctan

    # Return as tuple coordinate in decimal degrees
    destination = (np.rad2deg(destination_longitude_rad.squeeze()),
                   np.rad2deg(destination_latitude_rad.squeeze()))

    return destination


def reciprocal_bearing(
    bearing: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Return the reciprocal (back) bearing of the input bearing.

    Args:
        bearing (float | np.array): Bearing in decimal degrees.

    Returns:
        float | np.array: Back bearing(s) (modulo 360 degrees).
    """
    return (bearing + 180) % 360


def lonlat_to_xy(
    longitude: np.ndarray,
    latitude: np.ndarray,
    origin: Tuple
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a longitude-latitude grid to a cartesian coordinate system.

    Transform a longitude-latitude grid to a local cartesian coordinate
    system using a Transverse Mercator projection. Inputs should have shape
    `(x,y)` where `x` is the number of longitudes and `y` is the number of
    latitudes.  The `origin` is a tuple containing the longitude and latitude
    which define the center of the local grid (i.e., 0, 0), respectively.
    Returns an x-y meshgrid in meters.

    Note: Assumes a WGS84 datum and ellipsoid.

    Args:
        longitude (np.ndarray): longitude meshgrid with shape (x,y)
        latitude (np.ndarray):  latitude meshgrid with shape (x,y)
        origin (Tuple): longitude and latitude which define the origin of the
            local grid as (longitude_0, latitude_0).

    Returns:
        Tuple[np.ndarray, np.ndarray]: meshed x- and y-grids each with
            shape (x,y) and units of meters.
    """
    transformer = lonlat_tmerc_transformer(origin)
    x_grid, y_grid = transformer.transform(longitude, latitude)
    return x_grid, y_grid


def lonlat_tmerc_transformer(origin: Tuple) -> pyproj.Transformer:
    """Initialize a pyproj Transformer to convert longitude and latitude to xy.

    Initialize and return a pyproj Transformer object which converts longitude
    and latitude arrays to a local xy-coordinate system centered on `origin`
    using a Transverse Mercator (re)projection. The `origin` is a tuple
    containing the longitude and latitude which define the center of the local
    grid (i.e., 0, 0), respectively.

    Note: Assumes a 'WGS84' datum and ellipsoid.

    Args:
        origin (Tuple): longitude and latitude which define the origin of the
            local grid as (longitude_0, latitude_0).

    Returns:
        pyproj.Transformer: a 'latlong' to 'tmerc' transformer object.
    """
    proj_from = pyproj.Proj(proj='latlong', datum='WGS84')
    proj_to = pyproj.Proj(proj='tmerc', lon_0=origin[0], lat_0=origin[1], ellps='WGS84', preserve_units=True)
    return Transformer.from_proj(proj_from=proj_from, proj_to=proj_to)
