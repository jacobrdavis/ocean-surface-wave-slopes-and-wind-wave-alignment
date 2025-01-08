"""
Shared plotting functions.
"""


from typing import Optional, Union, Tuple, Sequence, Any

import cartopy
import cmocean
import colorcet
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh, PathCollection, LineCollection
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.contour import QuadContourSet, ContourSet
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyArrow, Arc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src import utilities

# Default drifter colors and labels
default_drifter_colors = {
    'microswift': 'rebeccapurple',  #  'grey',
    'spotter': 'goldenrod',
}
default_drifter_labels = {
    'microswift': 'MicroSWIFT',
    'spotter': 'Spotter',
}


def truncate_colormap(
    cmap: mpl.colors.LinearSegmentedColormap,
    minval: float = 0.0,
    maxval: float = 1.0,
    n: int = 256,
) -> mpl.colors.LinearSegmentedColormap:
    """
    Helper function to use a subset of a pre-defined colormap range

    Arguments:
        - cmap (colors.LinearSegmentedColormap), colormap to truncate
        - minval (float, optional), normalized min (1.0 is the bottom end of the full range); defaults to 0.0.
        - maxval (float, optional), normalized max (1.0 is the top end of the full range); defaults to 1.0.
        - n (int, optional), number of discrete colors; defaults to 256.

    Returns:
        - (colors.LinearSegmentedColormap), the truncated colormap

    Example:

    Return a colormap equivalent to the upper %70 of the 'Blues' colormap:
    >>> cmap = plt.get_cmap('Blues')
    >>> truncate_colormap(cmap, 0.3, 1)

    """
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval},{maxval})',
        cmap(np.linspace(minval, maxval, n)),
        N=n,
    )
    return new_cmap


# Default colormap and norm by variable
time_plot_kwargs = {
    'cmap': colorcet.cm.bgy,
    'norm': None,
}
mean_wave_age_plot_kwargs = {
    'cmap': 'Spectral_r',
    'norm': mpl.colors.LogNorm(vmin=0.25, vmax=3),
}
mean_inverse_wave_age_plot_kwargs = {
    'cmap': truncate_colormap(colorcet.cm.bmy, 0.10, 0.90, 256),
    'norm': mpl.colors.Normalize(vmin=1, vmax=4),
}
depth_plot_kwargs = {
    'cmap': cmocean.cm.deep,
    'norm': mpl.colors.Normalize(vmin=10, vmax=40)
}
relative_depth_plot_kwargs = {
    'cmap': cmocean.cm.deep,
    # 'norm': mpl.colors.LogNorm(vmin=np.pi/10, vmax=np.pi)
    'norm': mpl.colors.Normalize(vmin=np.pi/10, vmax=np.pi)
}
bathy_plot_kwargs = {
    # 'cmap': cmocean.tools.crop_by_percent(cmocean.cm.deep_r, 25, which='min', N=None),
    # 'cmap': colorcet.cm.blues_r,
    # 'cmap': truncate_colormap(colorcet.cm.blues_r, maxval=0.8),
    'cmap': truncate_colormap(mpl.cm.Blues_r, minval=0.05, maxval=0.75),
    # 'cmap': truncate_colormap(colorcet.cm.kbc, 0.25),
    'norm': mpl.colors.Normalize(vmin=-100, vmax=0),
    # 'norm': mpl.colors.Normalize(vmin=-50, vmax=0),
    'extend': 'min',
}
drift_speed_plot_kwargs = {
    'cmap': cmocean.cm.speed,
    'norm': mpl.colors.Normalize(vmin=0, vmax=2),
}
projected_drift_speed_plot_kwargs = {
    'cmap': colorcet.cm.bky,
    'norm': mpl.colors.Normalize(vmin=-1, vmax=1),
}
directional_spread_plot_kwargs = {
    'cmap': colorcet.cm.bmw,
    'norm': mpl.colors.Normalize(vmin=0, vmax=90),
}
wind_speed_plot_kwargs = {
    'cmap': truncate_colormap(colorcet.cm.bmy, 0, 0.90, 256),
    'norm': mpl.colors.Normalize(vmin=10, vmax=50),
}
pressure_plot_kwargs = {
    'cmap': 'Spectral_r',
    'norm': mpl.colors.Normalize(vmin=940, vmax=1020),
}
mean_square_slope_plot_kwargs = {
    'cmap': colorcet.cm.dimgray_r,
    'norm': mpl.colors.Normalize(vmin=0.010, vmax=0.022),
}
mean_square_slope_residual_plot_kwargs = {
    'cmap': cmocean.cm.balance,
    'norm': mpl.colors.Normalize(vmin=-0.004, vmax=0.004),
}
wind_wave_alignment_signed_plot_kwargs = {
    'cmap': mpl.cm.twilight,
    'norm': mpl.colors.Normalize(vmin=-180, vmax=180),
}
wind_wave_alignment_abs_plot_kwargs = {
    'cmap': cmocean.cm.balance,
    'norm': mpl.colors.Normalize(vmin=0, vmax=180),
}
alignment_cat_map = {
    'aligned': 0,
    'crossing': 1,
    'opposing': 2,
}
alignment_cat_map_inv = {num: cat for cat, num in alignment_cat_map.items()}
alignment_cat_cmap = plt.cm.get_cmap('coolwarm', 3)
alignment_cat_cmap = alignment_cat_cmap(range(3))
alignment_cat_cmap[1] = [0.6, 0.6, 0.6, 1.]
alignment_cat_cmap = LinearSegmentedColormap.from_list('coolwarm_new', alignment_cat_cmap, 3)
wind_wave_alignment_cat_plot_kwargs = {
    'cmap': alignment_cat_cmap,
    'norm': mpl.colors.Normalize(vmin=-0.5, vmax=2.5),
}


# Helper functions
def _set_kwarg_defaults(
    default_kwargs: dict,
    kwargs: Optional[dict] = None,
) -> dict:
    """ Set default keyword arguments for plotting functions."""
    if kwargs is None:
        kwargs = default_kwargs
    else:
        for key in default_kwargs.keys():
            if key not in kwargs:
                kwargs[key] = default_kwargs[key]
    return kwargs


def get_drifter_color(drifter_type: str) -> str:
    """ Return drifter color """
    return default_drifter_colors[drifter_type]


def get_drifter_label(drifter_type: str) -> str:
    """ Return drifter label """
    return default_drifter_labels[drifter_type]


# Plot layout
figure_full_width = 5.5
normal_font_size = 10
small_font_size = 8
default_gridline_kwargs = dict(
    color='k',
    alpha=0.075,
    linestyle='-',
    linewidth=0.5,
)


def configure_figures():
    """ Configure figure rc params.  """

    rc_params = {
        'font.size': normal_font_size,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Helvetica',
        'axes.titlesize': normal_font_size,
        'axes.linewidth': 0.5,
        'axes.labelsize': normal_font_size,
        'lines.markersize': 3,
        'legend.fontsize': small_font_size,
        'xtick.labelsize': small_font_size,
        'ytick.labelsize': small_font_size,
        'figure.dpi': 300,
        'figure.figsize': (figure_full_width, 4.125),
    }
    plt.rcParams.update(rc_params)


def set_square_aspect(ax):
    """ Set the aspect ratio of the axes to be square. """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    aspect = (xlim[1]-xlim[0]) / (ylim[1]-ylim[0])
    ax.set_aspect(aspect, adjustable='box')


def create_inset_colorbar(plot_handle, ax, bounds=None, **kwargs):
    """ Create an inset colorbar. """
    # bounds = [x0, y0, width, height]
    if bounds is None:
        bounds = [0.93, 0.5, 0.02, 0.45]
    cax = ax.inset_axes(bounds, axes_class=Axes)
    cbar = plt.colorbar(plot_handle, cax=cax, **kwargs)
    return cbar, cax


def set_gridlines(ax, **kwargs):
    """ Set axis gridlines. """
    grid_kwargs = _set_kwarg_defaults(default_gridline_kwargs, kwargs)
    ax.grid(**grid_kwargs)


def remove_top_and_right_spines(ax):
    """ Remove the top and right spines from an axis. """
    ax.spines[['right', 'top']].set_visible(False)


def get_empty_legend_placeholder() -> Line2D:
    """ Return an empty legend placeholder. """
    return mpl.lines.Line2D([], [], color="none")


# Subplot functions and classes
class SubplotLabeler:
    """ Create an object to label subplots. """
    def __init__(self):
        # self.ax = ax
        self.count = 0

    def increment_counter(self):
        self.count += 1

    def add_label(self, ax, **kwargs):
        label_letter = chr(ord('@') + (self.count % 26 + 1))
        text = f'({label_letter.lower()})'
        label_subplot(ax, text, **kwargs)
        self.increment_counter()


def label_subplot(
    ax,
    text,
    fontsize=normal_font_size,
    loc='upper left',
    nudge_x=0,
    nudge_y=0,
    **kwargs,
):
    """ Add text to subplot in the specified location. """
    if loc == 'upper left':
        xy = (0.05 + nudge_x, 0.95 + nudge_y)
        ha = 'left'
        va = 'top'
    elif loc == 'upper right':
        xy = (0.95 + nudge_x, 0.95 + nudge_y)
        ha = 'right'
        va = 'top'

    ax.annotate(
        text=text,
        xy=xy,
        xycoords='axes fraction',
        ha=ha,
        va=va,
        fontsize=fontsize,
        **kwargs,
    )


def axes_to_iterator(axes):
    """ Convert a 2D array of axes to an iterator. """
    return iter(axes.ravel())


# Default scatter plot marker keyword arguments
microswift_scatter_kwargs = {
    'color': default_drifter_colors['microswift'],
    'edgecolor': 'none',
    's': 3,
    'marker': 'o',
}
spotter_scatter_kwargs = {
    'color': default_drifter_colors['spotter'],
    'edgecolor': 'none',
    's': 3,
    'marker': 'o',
}

drifter_time_series_kwargs = dict(
    color=get_drifter_color('spotter'),
    marker='.',
    markersize=2,
    linewidth=2,
    alpha=0.35,
)
coamps_time_series_kwargs = dict(
    color='k',
    marker='.',
    markersize=2,
    linewidth=2,
    alpha=0.35,
)


# Drifter plotting
def plot_drifter_scatter(
    drifter_df: pd.DataFrame,
    ax: Axes,
    x_column_name: str,
    y_column_name: str,
    color_column_name: Optional[str] = None,
    **kwargs,
) -> PathCollection:
    """ Plot drifter data as a scatter plot. """
    if color_column_name is not None:
        color_column = drifter_df[color_column_name]
    else:
        color_column = None

    sc_plot = ax.scatter(
        drifter_df[x_column_name],
        drifter_df[y_column_name],
        c=color_column,
        **kwargs
    )
    return sc_plot


def plot_drifter_time_series_scatter(
    drifter_df: pd.DataFrame,
    ax: Axes,
    y_column_name: str,
    color_column_name: Optional[str] = None,
    **kwargs,
) -> PathCollection:
    """ Plot drifter time series data as a time series scatter plot. """
    if color_column_name is not None:
        color_column = drifter_df[color_column_name]
    else:
        color_column = None

    sc_plot = ax.scatter(
        drifter_df.index,
        drifter_df[y_column_name],
        c=color_column,
        **kwargs
    )
    return sc_plot


def plot_drifter_time_series_line(
    drifter_df: pd.DataFrame,
    ax: Axes,
    y_column_name: str,
    **kwargs,
) -> list[Line2D]:
    """ Plot drifter time series data as a time series line plot. """
    plot = ax.plot(
        drifter_df.index,
        drifter_df[y_column_name],
        **kwargs
    )
    return plot


def set_time_series_xaxis(
    ax: Axes,
    plot_time_start: pd.Timestamp,
    plot_time_end: pd.Timestamp,
    freq: str = '12h',
    format: str = '%m-%d %HZ',
) -> None:
    """ Format time series date axis. """
    date_ticks = pd.date_range(plot_time_start, plot_time_end, freq=freq)
    ax.set_xticks(date_ticks)
    ax.set_xlim([plot_time_start, plot_time_end])
    date_format = mdates.DateFormatter(format)
    ax.xaxis.set_major_formatter(date_format)


def lineplot_color(
    X: np.ndarray,
    Y: np.ndarray,
    z: np.ndarray,
    ax: Optional[Axes] = None,
    **kwargs,
) -> LineCollection:
    """ Plot a lines with a color map.

    Args:
        X (np.ndarray): x-values with shape (k, i)
        Y (np.ndarray): y-values with shape (k, j)
        c (np.ndarray): values mapped to colors with shape (k,)
        ax (Optional[Axes], optional): axes to plot on. Defaults to None.

    Returns:
        LineCollection: line plot colored based on `c`.
    """
    if ax is None:
        ax = plt.gca()

    # Add the arrays as a line collection.
    lines = np.dstack((X, Y))
    line_collection = LineCollection(lines, array=z, **kwargs)
    ax.add_collection(line_collection)

    return line_collection


# Default map marker keyword arguments
microswift_map_kwargs = {
    'color': default_drifter_colors['microswift'],
    'edgecolor': 'k',
    'linewidth': 0.5,
    's': 30,
    'marker': 'o',
    'zorder':  5,
}
spotter_map_kwargs = {
    'color': default_drifter_colors['spotter'],
    'edgecolor': 'k',
    'linewidth': 0.5,
    's': 35,
    'marker': 'p',
    'zorder': 5,
}


# Drifter mapping
def plot_drifter_track(
    drifter_df: pd.DataFrame,
    ax: GeoAxes,
    color_column_name: Optional[str] = None,
    first_only=True,
    **kwargs,
) -> PathCollection:
    """ Plot drifter track on a map. """
    if first_only:
        drifter_df_plot = get_multiindex_first(drifter_df)
    else:
        drifter_df_plot = drifter_df

    if color_column_name:
        color_column = drifter_df_plot[color_column_name]
    else:
        color_column = None

    plot = ax.scatter(drifter_df_plot['longitude'],
                      drifter_df_plot['latitude'],
                      c=color_column,
                      **kwargs)
    return plot


def update_drifter_track(
    drifter_df: pd.DataFrame,
    plot: PathCollection,
    color_column_name: Optional[str] = None,
) -> PathCollection:
    """ Update a previously mapped drifter track. """
    drifter_df_first = get_multiindex_first(drifter_df)
    xy_data = drifter_df_first[['longitude', 'latitude']].values
    if xy_data.size > 0:
        plot.set_offsets(xy_data)
    else:
        xy_data = []
    if color_column_name:
        color_column = drifter_df_first[color_column_name]
        plot.set_array(color_column.ravel())
    else:
        pass
    return plot


def plot_drifter_track_line(
    drifter_df: pd.DataFrame,
    ax: GeoAxes,
    **kwargs,
) -> PathCollection:
    """ Plot drifter track on a map. """
    plot = ax.plot(drifter_df['longitude'],
                   drifter_df['latitude'],
                   **kwargs)
    return plot


def plot_drifter_storm_frame(
    drifter_df: pd.DataFrame,
    ax: Axes,
    color_column_name=None,
    first_only=True,
    normalize_by_rmw=True,
    **kwargs,
) -> PathCollection:
    """ Plot drifter track on a storm-frame map. """
    if first_only:
        drifter_df_plot = get_multiindex_first(drifter_df)
    else:
        drifter_df_plot = drifter_df

    if color_column_name:
        color_column = drifter_df_plot[color_column_name]
    else:
        color_column = None

    # Get storm x- and y-distances (in km)
    storm_distance_x = drifter_df_plot['storm_distance_x']
    storm_distance_y = drifter_df_plot['storm_distance_y']

    if normalize_by_rmw:
        rmw_nmi = drifter_df_plot['storm_radius_max_wind_nmi']
        rmw_km = utilities.nmi_to_km(rmw_nmi)
        storm_distance_x = storm_distance_x / rmw_km
        storm_distance_y = storm_distance_y / rmw_km

    plot = ax.scatter(storm_distance_x,
                      storm_distance_y,
                      c=color_column,
                      **kwargs)
    return plot


def update_drifter_storm_frame(
    drifter_df: pd.DataFrame,
    plot: PathCollection,
    color_column_name: Optional[str] = None,
) -> PathCollection:
    """ Update a previously storm-frame mapped drifter track. """
    drifter_df_first = get_multiindex_first(drifter_df)
    xy_data = drifter_df_first[['storm_distance_x', 'storm_distance_y']].values
    if xy_data.size > 0:
        plot.set_offsets(xy_data)
    else:
        xy_data = []
    if color_column_name:
        color_column = drifter_df_first[color_column_name]
        plot.set_array(color_column.ravel())
    else:
        pass
    return plot


def annotate_drifter(drifter_df: pd.DataFrame, ax: Axes, **annotate_kwargs) -> None:
    """ Annotate a drifter with its ID. """
    drifter_df_first = get_multiindex_first(drifter_df)
    label_data = drifter_df_first.index.get_level_values(0).values
    xy_data = drifter_df_first[['longitude', 'latitude']].values
    for label, xy in zip(label_data, xy_data):
        annotation = ax.annotate(
            label,
            xy=xy,
            zorder=10,
            annotation_clip=True,
            **annotate_kwargs
        )


def get_multiindex_first(drifter_df, level=0):
    """ Return the first row of a multiindex DataFrame. """
    return drifter_df.groupby(level=level).first()


# Mapping

# Default cartopy map keyword arguments
default_ocean_kwargs = {'color': 'white'}
default_land_kwargs = {'color': 'whitesmoke', 'zorder': 3, 'alpha': 0.4}
default_coast_kwargs = {'edgecolor': 'grey', 'linewidth': 0.5, 'zorder': 4}
default_intensity_cmap = mpl.cm.get_cmap('YlOrRd', 7)
default_map_gridline_kwargs = dict(
    draw_labels=True,
    dms=False,
    x_inline=False,
    y_inline=False,
    zorder=1,
    alpha=0.25,
    xlabel_style={'size': small_font_size},
    ylabel_style={'size': small_font_size},
)


def plot_base_chart(
    ax: GeoAxes,
    extent: np.ndarray,
    ocean_kwargs=default_ocean_kwargs,
    land_kwargs=default_land_kwargs,
    coast_kwargs=default_coast_kwargs,
    gridline_kwargs=default_map_gridline_kwargs,
):
    """ Plot a base map features (ocean, land, coast, grid). """
    # Set remaining kwargs to defaults
    ocean_kwargs = _set_kwarg_defaults(default_ocean_kwargs, ocean_kwargs)
    land_kwargs = _set_kwarg_defaults(default_land_kwargs, land_kwargs)
    coast_kwargs = _set_kwarg_defaults(default_coast_kwargs, coast_kwargs)
    gridline_kwargs = _set_kwarg_defaults(default_map_gridline_kwargs, gridline_kwargs)

    # Initialize the figure, crop it based on extent, and add gridlines
    ax.set_extent(extent)
    ax.set_aspect('equal')
    gridlines = ax.gridlines(**gridline_kwargs)
    gridlines.top_labels = False
    gridlines.left_labels = False
    gridlines.right_labels = True

    # Add the ocean, land, coastline, and border features
    ax.add_feature(cartopy.feature.OCEAN, **ocean_kwargs)
    ax.add_feature(cartopy.feature.LAND, **land_kwargs)
    ax.add_feature(cartopy.feature.COASTLINE, **coast_kwargs)


default_inset_axes_kwargs = {
    'width': '30%',
    'height': '30%',
    'loc': 'upper right',
    'borderpad': 0,
    'axes_class': cartopy.mpl.geoaxes.GeoAxes,
    'axes_kwargs': dict(projection=cartopy.crs.PlateCarree()),
    #TODO:
}


def plot_minimap(
    ax: GeoAxes,
    extent: np.ndarray,
    inset_axes_kwargs=default_inset_axes_kwargs,
    ocean_kwargs=default_ocean_kwargs,
    land_kwargs=default_land_kwargs,
    coast_kwargs=default_coast_kwargs,
) -> GeoAxes:
    """ Plot an inset minimap. """
    #TODO: _update_default_kwargs...
    # Set remaining kwargs to defaults
    inset_axes_kwargs = _set_kwarg_defaults(default_inset_axes_kwargs, inset_axes_kwargs)
    ocean_kwargs = _set_kwarg_defaults(default_ocean_kwargs, ocean_kwargs)
    land_kwargs = _set_kwarg_defaults(default_land_kwargs, land_kwargs)
    coast_kwargs = _set_kwarg_defaults(default_coast_kwargs, coast_kwargs)

    ax_inset = inset_axes(ax, **inset_axes_kwargs)
    ax_inset.set_extent(extent)
    ax_inset.set_aspect('equal')

    # Add the ocean, land, coastline, and border features
    ax_inset.add_feature(cartopy.feature.OCEAN, **ocean_kwargs)
    ax_inset.add_feature(cartopy.feature.LAND, **land_kwargs)
    ax_inset.add_feature(cartopy.feature.COASTLINE, **coast_kwargs)

    return ax_inset


default_hurricane_offset_image_kwargs = {
    'zoom': 0.06,
}


def plot_hurricane_symbol(
    ax: Axes,
    image_path: str = './images/hurricane.png',
    xy: Tuple = (0, 0),
    offset_image_kwargs: Optional[dict] = None,
    annotation_bbox_kwargs: Optional[dict] = None,
) -> None:
    """ Plot a hurricane symbol on an axis. """
    offset_image_kwargs = _set_kwarg_defaults(
        default_hurricane_offset_image_kwargs,
        offset_image_kwargs,
    )
    image = plt.imread(image_path)
    plot_image(
        ax,
        image,
        xy,
        offset_image_kwargs=offset_image_kwargs,
        annotation_bbox_kwargs=annotation_bbox_kwargs,
    )


# COAMPS plotting
def plot_coamps_field(
    coamps_ds: xr.Dataset,
    field_name: str,
    ax: GeoAxes,
    **kwargs,
) -> QuadMesh:
    """ Plot a COAMPS field on a map. """
    x = coamps_ds['longitude'].values
    y = coamps_ds['latitude'].values
    field = coamps_ds[field_name].values
    X, Y = np.meshgrid(x, y)
    return ax.pcolormesh(X, Y, field, **kwargs)


def plot_coamps_field_storm_frame(
    coamps_ds: xr.Dataset,
    field_name: str,
    ax: GeoAxes,
    **kwargs,
) -> QuadMesh:
    """ Plot a COAMPS field on a storm-frame map. """
    x = coamps_ds['x'].values
    y = coamps_ds['y'].values
    field = coamps_ds[field_name].values
    if x.ndim == 1 and y.ndim == 1:
        x, y = np.meshgrid(x, y)
    return ax.pcolormesh(x, y, field, **kwargs)


def update_coamps_field(
    coamps_ds: np.ndarray,
    field_name: str,
    plot: QuadMesh,
) -> QuadMesh:
    """ Update a previously mapped COAMPS field (used in animation). """
    field = coamps_ds[field_name].values
    if field.size > 0:
        plot.set_array(field.ravel())
    else:
        plot.set_array([])
    return plot


# Bathymetry plotting
bathy_contour_kwargs = {
    'linestyles': '-',
    # 'linewidths': 1,
    'linewidths': 0.25,
    'alpha':0.5,
    # 'colors': 'none',
    'colors': 'grey',
}


def plot_bathymetry(
    bathy_ds: xr.Dataset,
    levels: list,
    ax: GeoAxes,
    **kwargs,
) -> QuadContourSet:
    """ Plot a bathymetry grid on a map. """
    bathy = ax.contourf(
        bathy_ds['lon'],
        bathy_ds['lat'],
        bathy_ds['elevation'],
        levels=levels,
        **kwargs,
    )

    return bathy


def plot_bathymetry_contours(
    bathy_ds: xr.Dataset,
    label_levels,
    label_locations,
    angle,
    ax: GeoAxes,
    fontsize=small_font_size,
    **kwargs,
) -> Tuple[QuadContourSet, ContourSet]:
    """ Plot bathymetry contours on a map. """
    bathy_contours = ax.contour(
        bathy_ds['lon'],
        bathy_ds['lat'],
        bathy_ds['elevation'],
        levels=label_levels,
        **kwargs,
    )

    def fmt_level_labels(level):
        level_abs = abs(level)
        label = f"{level_abs:.0f}"
        max_level_abs = max(np.abs(label_levels))
        if level_abs >= max_level_abs:
            return rf"{label}+ m"
        else:
            return rf"{label} m"

    contour_labels = ax.clabel(
        bathy_contours,
        colors='k',
        fontsize=fontsize,
        levels=label_levels,
        manual=label_locations,
        fmt=fmt_level_labels,
    )

    for l in contour_labels:
        l.set_rotation(angle)

    return bathy_contours, contour_labels




# Best track plotting
default_pts_kwargs = dict(
    edgecolor='k',
    zorder=4,
    markersize=175,
    linewidth=0.5,
    alpha=1.0,
)
default_annotate_pts_kwargs = dict(
    annotation_clip=True,
    zorder=10,
    fontsize=small_font_size,
    bbox=dict(boxstyle='circle,pad=0', fc='none', ec='none')
)


def plot_best_track(
    ax: GeoAxes,
    pts_gdf: Optional[gpd.GeoDataFrame] = None,
    lin_gdf: Optional[gpd.GeoDataFrame] = None,
    windswath_gdf: Optional[gpd.GeoDataFrame] = None,
    intensity_cmap: Colormap = default_intensity_cmap,
    pts_kwargs: Optional[dict] = None,
    annotate_pts_kwargs: Optional[dict] = None,
) -> GeoAxes:
    """ Plot a hurricane best track on a map.  """
    pts_kwargs = _set_kwarg_defaults(default_pts_kwargs, pts_kwargs)
    annotate_pts_kwargs = _set_kwarg_defaults(default_annotate_pts_kwargs, annotate_pts_kwargs)
    if lin_gdf is not None:
        lin_gdf.plot(
            color='k',
            zorder=2,
            ax=ax
        )

    if windswath_gdf is not None:
        windswath_gdf[windswath_gdf['RADII'] == 64.0].plot(
            facecolor='dimgrey',
            alpha=0.4,
            ax=ax,
        )

        windswath_gdf[windswath_gdf['RADII'] == 50.0].plot(
            facecolor='darkgrey',
            alpha=0.4,
            ax=ax,
        )

        windswath_gdf[windswath_gdf['RADII'] == 34.0].plot(
            facecolor='lightgrey',
            alpha=0.4,
            ax=ax,
        )

    # Plot the best track points; color and label by intensity
    if pts_gdf is not None:
        pts_gdf.plot(
            column='saffir_simpson_int',
            cmap=intensity_cmap,
            vmin=-1.5,
            vmax=5.5,
            ax=ax,
            **pts_kwargs
        )

        for x, y, label in zip(pts_gdf.geometry.x, pts_gdf.geometry.y, pts_gdf['saffir_simpson_label']):
            ax.annotate(
                label,
                xy=(x, y),
                ha='center',
                va='center',
                **annotate_pts_kwargs,
            )

    return ax


# Image plotting
default_annotation_bbox_kwargs = {
    'xycoords': 'data',
    'frameon': False,
    'clip_on': True,
}
default_offset_image_kwargs = {
    'zoom': 1,
}


def plot_image(
    ax: Union[Axes, GeoAxes],
    image: np.ndarray,
    xy: Sequence[float],
    offset_image_kwargs: Optional[dict] = None,
    annotation_bbox_kwargs: Optional[dict] = None,
) -> None:
    """ Plot an image on an axis. """
    annotation_bbox_kwargs = _set_kwarg_defaults(
        default_annotation_bbox_kwargs,
        annotation_bbox_kwargs,
    )
    offset_image_kwargs = _set_kwarg_defaults(
        default_offset_image_kwargs,
        offset_image_kwargs,
    )
    imagebox = OffsetImage(image, **offset_image_kwargs)
    imagebox.image.axes = ax  #TODO: check if this is necessary
    annotation = AnnotationBbox(
        imagebox,
        xy=xy,
        **annotation_bbox_kwargs,
    )
    ax.add_artist(annotation)


def hist_log(x, n_bins=10, ax=None, **kwargs):
    """ Plot a histogram with log-spaced bins.
    See:
    https://stackoverflow.com/questions/47850202/plotting-a-histogram-
    on-a-log-scale-with-matplotlib
    """
    if ax is None:
        ax = plt.gca()
    _, bins = np.histogram(x, bins=n_bins)
    bins_log = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    ax.hist(
        x,
        bins=bins_log,
        **kwargs,
    )
