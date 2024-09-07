# Input data

To run this repository, the following files are required:

## Idalia drifter data

`hurricane_idalia_drifter_data_v3.pickle`

Spotter and MicroSWIFT drifter (buoy) data in Hurricane Idalia from 2023-08-28 to 2023-08-31.

This file should be downloaded and saved locally in this folder (`input_data/`) and the path should be stored in config.toml as `idalia_drifter_data`.
Available on the Dryad repository for this publication (https://doi.org/10.5061/dryad.zw3r228h7).

## COAMPS-TC data

`idalia_coamps_aggregated_forecast_winds_tau4_08-29_to_08-31.nc`

COAMPS-TC wind and pressure fields for Hurricane Idalia from 2023-08-29 to 2023-08-31.  Originally accessed from MetGet on October 6, 2023 at 1600 PT.

This file should be downloaded and saved locally in this folder (`input_data/`) and the path should be stored in config.toml as `coamps`.
Available on the Dryad repository for this publication (https://doi.org/10.5061/dryad.zw3r228h7).

## NHC best track shape files

`al102023_best_track/AL102023_pts*`

`al102023_best_track/AL102023_lin*`

`al102023_best_track/AL102023_radii*`

`al102023_best_track/AL102023_windswath*`

NHC shape files of best track points, line, radii, and wind swath.  Each component should include the `.shp` file and all supporting files (`.dbf`, `.prj`, `.shp.xml`, `.shx`).
These data were originally accessed from https://www.nhc.noaa.gov/data/tcr/index.php?season=2023&basin=atl on October 6, 2023 at 1600 PT.

This folder (and all of its files) should be downloaded and saved locally in this folder (`input_data/`) and the path should be stored in config.toml as `best_track`.
Available on the Dryad repository for this publication (https://doi.org/10.5061/dryad.zw3r228h7).

## IBTrACS

`ibtracs.last3years.list.v04r00.csv`

International Best Track Archive for Climate Stewardship (IBTrACS) tabular data from the last 3 years (used for storm metrics).  These data were originally accessed from https://www.ncei.noaa.gov/products/international-best-track-archive on January 4, 2024 at 1200 PT.

This file should be downloaded and saved locally in this folder (`input_data/`) and the path should be stored in config.toml as `ibtracs`.
Available on the Dryad repository for this publication (https://doi.org/10.5061/dryad.zw3r228h7).

## GEBCO

`gebco_2023_n31.0_s25.0_w-87.0_e-81.0.nc`

Gridded bathymetry data from the General Bathymetric Chart of the Oceans (GEBCO) GEBCO_2023 grid.  These data were originally accessed from https://www.gebco.net/data_and_products/gridded_bathymetry_data/ on November 28, 2023 at 1200 PT.

This file should be downloaded and saved locally in this folder (`input_data/`) and the path should be stored in config.toml as `gebco`.
Available on the Dryad repository for this publication (https://doi.org/10.5061/dryad.zw3r228h7).

## Ian drifter data

`ian_spotter_coamps_data.json`

Spotter drifter (buoy) data in Hurricane Ian from 2022-09-27 to 2022-09-30.

This file should be downloaded and saved locally in this folder (`input_data/`) and the path should be stored in config.toml as `ian_drifter_data`.
Available on the Dryad repository for Davis et al. (2023) (https://doi.org/10.5061/dryad.g4f4qrfvb).


## Fiona drifter data

`fiona_spotter_coamps_data.json`

Spotter drifter (buoy) data in Hurricane Fiona from 2022-09-20 to 2022-09-25.

This file should be downloaded and saved locally in this folder (`input_data/`) and the path should be stored in config.toml as `fiona_drifter_data`.
Available on the Dryad repository for Davis et al. (2023) (https://doi.org/10.5061/dryad.g4f4qrfvb).
