"""
Pandas Dataframe `buoy` accessor and associated methods.
"""


__all__ = [
    "BuoyDataFrameAccessor",
]


import warnings
from types import SimpleNamespace
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
from pandas.api.typing import DataFrameGroupBy

from src import utilities, kinematics, waves, list_helpers


# DataFrame columns (and indexes) are defined in namespace.toml.
namespace = utilities.get_namespace()
var_namespace = utilities.get_var_namespace()


@pd.api.extensions.register_dataframe_accessor("buoy")
class BuoyDataFrameAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._vars = var_namespace

    @property
    def vars(self) -> SimpleNamespace:
        """ Return a SimpleNamespace with this DataFrame's variable names. """
        return self._vars

    @property
    def spectral_variables(self) -> List:
        """ Return a list of spectral variables in the DataFrame. """
        return self._get_spectral_variables()[0]

    @property
    def uniform_frequency(self) -> bool:
        """ Return True if all frequency arrays have the same shape. """
        spectral_size_df = self._get_spectral_variables()[1]
        unique_spectral_sizes = spectral_size_df.apply(pd.unique, axis=0)
        return len(unique_spectral_sizes) == 1

    def _get_element_sizes(self) -> pd.DataFrame:
        """ Return a DataFrame of sizes for each element in the DataFrame. """
        # Apply np.size element-wise to generate a DataFrame of sizes
        return self._obj.map(np.size, na_action='ignore')

    def _get_spectral_variables(
        self,
        frequency_col: Optional[str] = None
    ) -> Tuple[List, pd.DataFrame]:
        """ Return a spectral variable names list and DataFrame of sizes. """
        if frequency_col is None:
            frequency_col = self.vars.frequency

        # Compare each column in size_df to the frequency column and return
        # only the matching columns, which should be spectral.
        size_df = self._get_element_sizes()
        try:
            is_spectral = size_df.apply(
                lambda col: size_df[frequency_col].equals(col)
            )
            spectral_variable_names = is_spectral.index[is_spectral].to_list()
            spectral_variable_sizes = size_df.loc[:, spectral_variable_names]
        except KeyError:
            spectral_variable_names = []
            spectral_variable_sizes = pd.DataFrame()
        return spectral_variable_names, spectral_variable_sizes

    def to_xarray(
        self,
        frequency_col: Optional[str] = None,
        time_col: Optional[str] = None,
        set_datetime_index: bool = True,
    ) -> xr.Dataset:
        """ Return this DataFrame as an Xarray Dataset. """
        if frequency_col is None:
            frequency_col = self.vars.frequency
        if time_col is None:
            time_col = self.vars.time

        # Bulk (single value) and spectral columns must be handled separately
        # since `.to_xarray()` does not convert elements containing arrays.
        spectral_variables = self._get_spectral_variables(frequency_col)[0]
        drifter_bulk_ds = (self._obj
                           .drop(columns=spectral_variables)
                           .to_xarray())
        if spectral_variables:
            drifter_spectral_ds = (self._obj
                                   .loc[:, spectral_variables]
                                   .explode(spectral_variables)
                                   .set_index(frequency_col, append=True)
                                   .to_xarray())
            # Sometimes spectral variable arrays are converted as objects.
            drifter_spectral_ds = drifter_spectral_ds.astype(float)
        else:
            warnings.warn(
                '\n'
                'No spectral variables found in DataFrame.  If you expected \n'
                'spectral variables in this Dataset, check that "frequency" \n'
                'is defined correctly in the buoy.vars namespace, or \n'
                'provide `frequency_col` as an argument.\n'
            )
            drifter_spectral_ds = xr.Dataset()

        drifter_ds = xr.merge([drifter_bulk_ds, drifter_spectral_ds])

        if set_datetime_index:
            drifter_ds[time_col] = pd.DatetimeIndex(drifter_ds[time_col].values)

        return drifter_ds

    def to_netcdf(
        self,
        buoy_id: str,
        path: Optional[str] = None,
        ncdf_attrs_key: str = 'ncdf_attrs',
        float_fill_value: float = np.nan,
        frequency_col: Optional[str] = None,
        time_col: Optional[str] = None,
        **kwargs,
    ) -> xr.Dataset:
        """Save a DataFrame of a single buoy as a netCDF dataset.

        Convert a single `buoy_id` to an CF-compliant xarray Dataset using the
        attributes defined under the `ncdf_attrs_key` and save it to
        `path` as an netCDF dataset. The dataset is indexed using a POSIX time
        array, following the CF Conventions a "single trajectory". Default
        attributes follow CF-1.11 and ACDD-1.3 and use standard names from the
        CF Standard Name Table Version 85, (21 May 2024).

        Note:
            Only variables with attributes in `ncdf_attrs` are included. The
            DataFrame should NOT be multiindexed for compliance with the
            "single trajectory" CF conventions. Remaining kwargs are passed to
            xarray.to_netcdf().

        See also:
        https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#_single_trajectory

        Args:
            buoy_id (str): The buoy ID to save as a netCDF file.  If the
                DataFrame is multiindexed by id, this should be a valid index.
            path (Optional[str], optional): NetCDF path (including '.nc').
                If path is None, the Dataset is not saved. Defaults to None.
            ncdf_attrs_key (str, optional): TOML namespace key
                containing a dictionary of netCDF attributes.  This should be a
                subtable of 'buoy'.  Defaults to 'ncdf_attrs'.
            float_fill_value (float, optional): Null data fill value to use in
                float arrays. Defaults to np.nan.
            frequency_col (Optional[str], optional): DataFrame column name
                to use as the frequency coordinate. Defaults to None which is
                replaced with vars.frequency.
            time_col (Optional[str], optional): DataFrame column name
                to use as the time coordinate. Defaults to None which is
                replaced with vars.time.

        Returns:
            xr.Dataset: CF-compliant xarray Dataset.
        """
        # TODO: rename then to xarray
        if frequency_col is None:
            frequency_col = self.vars.frequency
        if time_col is None:
            time_col = self.vars.time

        # Get variable namespace and netCDF attributes from namespace.toml.
        var_namespace_dict = self._vars.__dict__
        # TODO: get ncdf attrs?
        ncdf_attrs = namespace['buoy'][ncdf_attrs_key]

        # Intersect variable namespace keys with the netCDF attribute keys.
        ncdf_var_keys = list_helpers.list_intersection(
            list_a=list(var_namespace_dict.keys()),
            list_b=list(ncdf_attrs.keys())
        )

        # Export variables that have netCDF attributes (excluding the indexes).
        vars_to_export = [var_namespace_dict[key] for key in ncdf_var_keys]
        vars_to_export = list_helpers.list_difference(
            list_a=vars_to_export,
            list_b=list(self._obj.index.names)
        )

        # If the DataFrame is multiindexed by id, select the buoy_id.
        if self.vars.id in self._obj.index.names:
            ncdf_df = self._obj.loc[buoy_id]
        else:
            ncdf_df = self._obj.copy()

        # Convert the DataFrame to an xarray Dataset with a POSIX index.
        ncdf_ds = (ncdf_df
                   .loc[:, vars_to_export]  # .loc[buoy_id, vars_to_export]
                   .set_index(_datetimeindex_to_posix(ncdf_df.index))
                   .buoy.to_xarray(frequency_col=frequency_col,
                                   time_col=time_col,
                                   set_datetime_index=False))

        # Assign netCDF attributes.
        for var_key, var_name in var_namespace_dict.items():
            if var_key in ncdf_attrs.keys():
                ncdf_ds[var_name].attrs = ncdf_attrs[var_key]
            else:
                pass

        # Assign the fill value to all float variables.
        float_vars = [var for var, dtype in ncdf_ds.dtypes.items() if np.issubdtype(dtype, np.floating)]
        ncdf_ds[float_vars] = ncdf_ds[float_vars].fillna(float_fill_value)

        # Create a trajectory DataArray per the CF Conventions for a single trajectory.
        trajectory_da = xr.DataArray(
            name='trajectory',
            data=buoy_id,
            attrs={'cf_role': "trajectory_id",
                   'long_name': 'buoy identifier'}
        )

        # Merge the trajectory DataArray with Dataset (this ensures it appears first).
        ncdf_ds = xr.merge([xr.Dataset({'trajectory': trajectory_da}), ncdf_ds],
                           combine_attrs='no_conflicts')

        # Add user-defined global attributes
        ncdf_ds.attrs = ncdf_attrs['global']

        # Add global CF attributes.
        current_iso_time_str = pd.Timestamp.now().isoformat(timespec='seconds')
        ncdf_ds.attrs['history'] = current_iso_time_str + ". Created."
        ncdf_ds.attrs['_FillValue'] = float_fill_value

        # Add global ACDD recommended attributes.
        #TODO: use vars namespace for latitude and longitude and time
        ncdf_ds.attrs['date_created'] = current_iso_time_str
        ncdf_ds.attrs['geospatial_lat_min'] = f"{ncdf_ds['latitude'].values.min():0.4f}"
        ncdf_ds.attrs['geospatial_lat_max'] = f"{ncdf_ds['latitude'].values.max():0.4f}"
        ncdf_ds.attrs['geospatial_lon_min'] = f"{ncdf_ds['longitude'].values.min():0.4f}"
        ncdf_ds.attrs['geospatial_lon_max'] = f"{ncdf_ds['longitude'].values.max():0.4f}"
        ncdf_ds.attrs['time_coverage_start'] = pd.Timestamp(ncdf_ds['time'].values.min(), unit='s').isoformat()
        ncdf_ds.attrs['time_coverage_end'] = pd.Timestamp(ncdf_ds['time'].values.max(), unit='s').isoformat()

        if path is not None:
            ncdf_ds.to_netcdf(path, **kwargs)

        return ncdf_ds


    def bin_by(
        self,
        bin_col: str,
        bins: np.ndarray,
        **kwargs
    ) -> DataFrameGroupBy:
        """ Bin a dataframe by a column and return a groupby object. """
        return self._obj.groupby(pd.cut(self._obj[bin_col], bins), **kwargs)

    def row_average(self) -> pd.DataFrame:
        """ Return the mean row values for each column. """
        mean_df = self._obj.apply(
            lambda df: df.buoy.stacked_mean(axis=0),
            axis=0
        )
        return mean_df

    def frequency_to_wavenumber(
        self,
        frequency_col: Optional[str] = None,
        depth_col: Optional[str] = None,
        **kwargs
    ) -> pd.Series:
        """ Convert frequency to wavenumber and return it as a Series. """
        if frequency_col is None:
            frequency_col = self.vars.frequency
        if depth_col is None:
            depth_col = self.vars.depth

        # If depth data is present, use the full relationship. Otherwise, only
        # the deep water relationship can be used.
        if depth_col in self._obj.columns:
            wavenumber = self._obj.apply(
                lambda df: waves.inverse_dispersion(
                    df[frequency_col],
                    np.array([df[depth_col]]),
                    **kwargs
                ),
                axis=1,
            )
        else:
            wavenumber = self._obj.apply(
                lambda df: waves.deep_water_dispersion(
                    df[frequency_col],
                    **kwargs,
                ),
                axis=1,
            )
        return wavenumber

    def mean_square_slope(
        self,
        energy_density_col: Optional[str] = None,
        frequency_col: Optional[str] = None,
        **kwargs
    ) -> pd.Series:
        """ Calculate mean square slope and return it as a Series. """
        if energy_density_col is None:
            energy_density_col = self.vars.energy_density
        if frequency_col is None:
            frequency_col = self.vars.frequency

        mean_square_slope = self._obj.apply(
                lambda df: waves.mean_square_slope(
                    energy_density=df[energy_density_col],
                    frequency=df[frequency_col],
                    **kwargs,
                ),
                axis=1,
            )
        return mean_square_slope

    def wavenumber_mean_square_slope(
        self,
        energy_density_wn_col: Optional[str] = None,
        wavenumber_col: Optional[str] = None,
        **kwargs
    ) -> pd.Series:
        """
        Calculate wavenumber mean square slope and return it as a Series.
        """
        if energy_density_wn_col is None:
            energy_density_wn_col = self.vars.energy_density_wn
        if wavenumber_col is None:
            wavenumber_col = self.vars.wavenumber

        wavenumber_mean_square_slope = self._obj.apply(
                lambda df: waves.wavenumber_mean_square_slope(
                    energy_density_wn=df[energy_density_wn_col],
                    wavenumber=df[wavenumber_col],
                    **kwargs,
                ),
                axis=1,
            )
        return wavenumber_mean_square_slope

    def energy_period(
        self,
        energy_density_col: Optional[str] = None,
        frequency_col: Optional[str] = None,
        **kwargs
    ) -> pd.Series:
        """ Calculate energy-weighted period and return it as a Series. """
        if energy_density_col is None:
            energy_density_col = self.vars.energy_density
        if frequency_col is None:
            frequency_col = self.vars.frequency

        energy_period = self._obj.apply(
                lambda df: waves.energy_period(
                    energy_density=df[energy_density_col],
                    frequency=df[frequency_col],
                    **kwargs,
                ),
                axis=1,
            )
        return energy_period

    def wave_direction(
        self,
        a1_col: Optional[str] = None,
        b1_col: Optional[str] = None,
        **kwargs
    ) -> pd.Series:
        """
        Calculate wave direction per frequency and return it as a Series.
        """
        if a1_col is None:
            a1_col = self.vars.a1
        if b1_col is None:
            b1_col = self.vars.b1

        direction = self._obj.apply(
                lambda df: waves.direction(
                    df[a1_col],
                    df[b1_col],
                    **kwargs,
                ),
                axis=1,
            )
        return direction

    def wave_directional_spread(
        self,
        a1_col: Optional[str] = None,
        b1_col: Optional[str] = None,
        **kwargs
    ) -> pd.Series:
        """
        Calculate wave directional spread per frequency and return it as a
        Series.
        """
        if a1_col is None:
            a1_col = self.vars.a1
        if b1_col is None:
            b1_col = self.vars.b1

        directional_spread = self._obj.apply(
                lambda df: waves.directional_spread(
                    df[a1_col],
                    df[b1_col],
                    **kwargs,
                ),
                axis=1,
            )
        return directional_spread

    def moment_weighted_mean(
        self,
        column: str,
        n: int = 0,
        energy_density_col: Optional[str] = None,
        frequency_col: Optional[str] = None,
    ) -> pd.Series:
        """ Return the nth moment-weighted mean of a column as a Series. """
        if energy_density_col is None:
            energy_density_col = self.vars.energy_density
        if frequency_col is None:
            frequency_col = self.vars.frequency

        moment_weighted_mean_series = self._obj.apply(
            lambda df: waves.moment_weighted_mean(
                arr=df[column],
                energy_density=df[energy_density_col],
                frequency=df[frequency_col],
                n=n,
            ),
            axis=1,
        )
        return moment_weighted_mean_series

    def sliding_energy_weighted_direction(
        self,
        window_size: int,
        energy_density_col: Optional[str] = None,
        a1_col: Optional[str] = None,
        b1_col: Optional[str] = None,
        **kwargs
    ) -> pd.Series:
        """
        Return a smoothed, frequency-dependent direction by computing the
        energy-weighted direction in sliding windows.
        """
        if energy_density_col is None:
            energy_density_col = self.vars.energy_density
        if a1_col is None:
            a1_col = self.vars.a1
        if b1_col is None:
            b1_col = self.vars.b1

        smoothed_direction = self._obj.apply(
                lambda df: waves.sliding_energy_weighted_direction(
                    energy_density=df[energy_density_col],
                    a1=df[a1_col],
                    b1=df[b1_col],
                    window_size=window_size,
                    **kwargs,
                ),
                axis=1,
            )
        return smoothed_direction

    def drift_speed_and_direction(
        self,
        longitude_col: Optional[str] = None,
        latitude_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Calculate drift speed and direction and return in a DataFrame.

        Note: this function assumes that `longitude` and `latitude` are
        subsequent positions sorted by `time`.  A groupby operation may be
        needed if several buoys are concatenated in the DataFrame.

        If using a mutliindex, the `id` level should first be used to group
        for correct results.
        """
        if longitude_col is None:
            longitude_col = self.vars.longitude
        if latitude_col is None:
            latitude_col = self.vars.latitude
        if time_col is None:
            time_col = self.vars.time

        drift_speed_mps, drift_dir_deg = kinematics.drift_speed_and_direction(
            longitude=self._obj[longitude_col],
            latitude=self._obj[latitude_col],
            time=self._obj.index.get_level_values(level=time_col),
            append=True,
        )
        drift_df = pd.DataFrame(
            data={
                self.vars.drift_speed: drift_speed_mps,
                self.vars.drift_direction: drift_dir_deg,
            },
            index=self._obj.index
        )
        return drift_df

    def wavenumber_energy_density(
        self,
        energy_density_col: Optional[str] = None,
        frequency_col: Optional[str] = None,
        depth_col: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Convert frequency domain wave energy density to wavenumber domain
        energy density and return a DataFrame with the energy and wavenumber.
        """
        if energy_density_col is None:
            energy_density_col = self.vars.energy_density
        if frequency_col is None:
            frequency_col = self.vars.frequency
        if depth_col is None:
            depth_col = self.vars.depth

        wavenumber_energy_density = self._obj.apply(
            lambda df: waves.fq_energy_to_wn_energy(
                energy_density_fq=df[energy_density_col],
                frequency=df[frequency_col],
                depth=df[depth_col],
                **kwargs,
            ),
            result_type='expand',
            axis=1,
        )
        return wavenumber_energy_density

    def merge_frequencies(
        self,
        n_merge: int,
        spectral_cols: Optional[List[str]] = None,
        frequency_col: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Return spectral columns with `n` merged frequencies in a new DataFrame.
        """
        if frequency_col is None:
            frequency_col = self.vars.frequency
        if spectral_cols is None:
            spectral_cols = self.spectral_variables

        merged_spectra = []
        for col in spectral_cols:
            merged_spectrum = self._obj.apply(
                    lambda df: waves.merge_frequencies(
                        spectrum=df[col],
                        n_merge=n_merge,
                        **kwargs,
                    ),
                    axis=1,
                )
            merged_spectrum.name = col  #  + '_merged'
            merged_spectra.append(merged_spectrum)

        # Concatenate the columns into a single DataFrame.
        return pd.concat(merged_spectra, axis=1)

    def doppler_adjust(
        self,
        frequency_cutoff,
        energy_density_obs_col: Optional[str] = None,
        frequency_obs_col: Optional[str] = None,
        drift_speed_col: Optional[str] = None,
        drift_direction_col: Optional[str] = None,
        direction_col: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Doppler adjust a 1-D spectrum to the intrinsic reference frame
        using the omnidirectional solutions and return as a DataFrame.

        If using a mutliindex, the `id` level should first be used to group
        for correct results.
        """
        if energy_density_obs_col is None:
            energy_density_obs_col = self.vars.energy_density
        if frequency_obs_col is None:
            frequency_obs_col = self.vars.frequency
        if drift_speed_col is None:
            drift_speed_col = self.vars.drift_speed
        if drift_direction_col is None:
            drift_direction_col = self.vars.drift_direction
        if direction_col is None:
            direction_col = self.vars.direction

        # Apply the Doppler adjustment to each frequency array. Results can be
        # added directly to the DataFrame copy using `result_type='expand'`.
        # new_df[new_cols] = new_df.apply(
        intrinsic_spectrum = self._obj.apply(
            lambda df: kinematics.doppler_adjust(
                energy_density_obs=df[energy_density_obs_col],
                frequency_obs=df[frequency_obs_col],
                drift_speed=df[drift_speed_col],
                drift_direction_going=df[drift_direction_col],
                wave_direction_coming=df[direction_col],
                frequency_cutoff=frequency_cutoff,
                **kwargs,
            ),
            axis=1,
            result_type='expand',
        )
        new_cols = [self.vars.energy_density_intrinsic,
                    self.vars.frequency_intrinsic]
        intrinsic_spectrum.columns = new_cols
        return intrinsic_spectrum


@pd.api.extensions.register_series_accessor("buoy")
class BuoySeriesAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def stacked_mean(self, axis=0) -> np.ndarray:
        """ Stack rows and return the mean along an axis. """
        return np.nanmean(np.stack(self._obj.values), axis=axis)


def _datetimeindex_to_posix(datetimeindex: pd.DatetimeIndex) -> pd.Index:
    """
    Convert a DatetimeIndex to an Index of POSIX times (seconds since 00:00:00
    1-Jan-1970 UTC).
    """
    return datetimeindex.astype(np.int64) // 10 ** 9
