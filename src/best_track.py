"""
Best track functions.
"""


import geopandas as gpd
import numpy as np
import pandas as pd


SAFFIR_SIMPSON = {
    'D' : {'range': (0  , 38.99),  'int': -1},
    'TS': {'range': (39 , 73.99),  'int':  0},
    '1' : {'range': (74 , 95.99),  'int':  1},
    '2' : {'range': (96 , 110.99), 'int':  2},
    '3' : {'range': (111, 129.99), 'int':  3},
    '4' : {'range': (130, 156.99), 'int':  4},
    '5' : {'range': (157, 200),    'int':  5},
}


def read_shp_file(
    path: str,
    crs: str = "EPSG:4326",
    index_by_datetime: bool = False,
) -> gpd.GeoDataFrame:
    """ Read a shape file (.shp) into a GeoDataFrame.

    Read a shape file (.shp) into a GeoDataFrame and assign a  coordinate
    reference system (crs).  By default, the WGS84 (EPSG:4326) datum is used.

    Args:
        path (str): path to the shape file.
        crs (str, optional): cartopy coordinate reference system. Defaults to
            "EPSG:4326").
        index_by_datetime (bool, optional): if True, assign the datetime as the
            GeoDataFrame index. Defaults to True.

    Returns:
        gpd.GeoDataFrame: shape file data in the specified crs.
    """
    shp_gdf = gpd.read_file(path)
    if index_by_datetime:
        try:
            set_best_track_datetime_index(shp_gdf)
        except KeyError as error:
            print(f'Unable to set datetime index due to KeyError ({error}).')

    return shp_gdf.to_crs(crs)


def read_kml_file(path: str) -> gpd.GeoDataFrame:
    """ Read a KML file (.kml) into a GeoDataFrame.

    Read a KML file (.kml) into a GeoDataFrame.  The output coordinate
    reference system (crs) will be the same as what is specified in the KML.

    Args:
        path (str): path to the KML file.

    Returns:
        gpd.GeoDataFrame: KML file data.
    """
    # Enable fiona driver
    gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
    return gpd.read_file(path, driver='KML')


def set_best_track_datetime_index(best_track: pd.DataFrame) -> pd.DataFrame:
    """ Convert NHC best track timestamps to datetimes.

    Convert the time information in NHC best track shapefiles, stored in
    separate year, month, day, hour, and minute fields, into a unified
    datetime and set it as the index.

    Args:
        best_track (pd.DataFrame): best track as downloaded from NHC

    Returns:
        pd.DataFrame: DataFrame with unified datetime index
    """
    datetime_columns = ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']
    best_track = (
        best_track
        .assign(HOUR=lambda df: df['HHMM'].str[:2])
        .assign(MINUTE=lambda df: df['HHMM'].str[2:])
        .assign(datetime=lambda df: pd.to_datetime(df[datetime_columns], utc=True))
        .drop(datetime_columns + ['HHMM'], axis=1)
        .set_index('datetime', drop=True)
    )
    return best_track


def best_track_pts_to_intensity(pts_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """ Categorize best track points using  Saffir Simpson scale.

    Categorize the best track intensities (provided as wind speeds) using the
    Saffir Simpson scale. The input GeoDataFrame must have an 'INTENSITY'
    column. The output has two additional columns ('saffir_simpson_label' and
    'saffir_simpson_int') which are the Saffir Simpson scale labels
    ['D', 'TS', '1', ... '5'] and corresponding integers [-1, 0, 1, ... 5]
    (which are useful for colormapping).

    Args:
        pts_gdf (gpd.GeoDataFrame): NHC best track points

    Returns:
        gpd.GeoDataFrame: original GeoDataFrame with columns for Saffir Simpson
            label and intensity.
    """
    for cat, definition in SAFFIR_SIMPSON.items():
        range_kn = np.array(definition['range'])
        in_range = pts_gdf['INTENSITY'].between(*mph_2_knots(range_kn))
        pts_gdf.loc[in_range, 'saffir_simpson_label'] = cat
        pts_gdf.loc[in_range, 'saffir_simpson_int'] = definition['int']

    return pts_gdf


def mph_2_knots(mph):
    """ Helper function convert wind speeds from mph to knots. """
    return mph * 0.868976
