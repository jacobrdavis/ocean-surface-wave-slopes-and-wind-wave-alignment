# Input data

To run this repository, the following files are required:

## Idalia drifter data

`hurricane_idalia_drifter_data_v3.pickle`

Spotter and MicroSWIFT drifter (buoy) data in Hurricane Idalia from 2023-08-28 to 2023-08-31.

This file should be downloaded and saved locally in this folder (`input_data/`) and the path should be stored in config.toml as `idalia_drifter_data`.
Available on the Dryad repository for this publication (https://doi.org/10.5061/dryad.zw3r228h7).

## COAMPS-TC data

`coamps_idalia_reforecast.nc`

COAMPS-TC "reforecast" wind and pressure fields for Hurricane Idalia from 2023-08-29 to 2023-08-31.  Originally accessed from MetGet on 1 November 2024 at 1600 PT.

This file should be downloaded and saved locally in this folder (`input_data/`) and the path should be stored in config.toml as `coamps`.
Available on the Dryad repository for this publication (https://doi.org/10.5061/dryad.zw3r228h7).

## NHC best track shape files

`al102023_best_track/AL102023_pts*`

`al102023_best_track/AL102023_lin*`

`al102023_best_track/AL102023_radii*`

`al102023_best_track/AL102023_windswath*`

NHC shape files of best track points, line, radii, and wind swath.  Each component should include the `.shp` file and all supporting files (`.dbf`, `.prj`, `.shp.xml`, `.shx`).
These data were originally accessed from https://www.nhc.noaa.gov/data/tcr/index.php?season=2023&basin=atl on 11 November 2024.
These data are used in accordance with the NOAA/NWS Data and Products disclaimer at: https://www.weather.gov/disclaimer/.

This folder (and all of its files) should be downloaded and saved locally in this folder (`input_data/`) and the path should be stored in config.toml as `best_track`.
Available on the Dryad repository for this publication (https://doi.org/10.5061/dryad.zw3r228h7).

## IBTrACS

`ibtracs.last3years.list.v04r01.csv`

International Best Track Archive for Climate Stewardship (IBTrACS) tabular data from the last 3 years (used for storm metrics).  These data were originally accessed from https://www.ncei.noaa.gov/products/international-best-track-archive on 7 November 2024.
Terms of use are described under the "Support" tab at: https://www.ncei.noaa.gov/products/international-best-track-archive.

This file should be downloaded and saved locally in this folder (`input_data/`) and the path should be stored in config.toml as `ibtracs`.
Available on the Dryad repository for this publication (https://doi.org/10.5061/dryad.zw3r228h7).

## GEBCO

`gebco_2023_n31.0_s25.0_w-87.0_e-81.0.nc`

Gridded bathymetry data from the General Bathymetric Chart of the Oceans (GEBCO) GEBCO_2023 grid.  These data were originally accessed from https://www.gebco.net/data_and_products/gridded_bathymetry_data/ on 28 November 2023.
Terms of use are described under the "Terms of use and disclaimer" header at: https://www.gebco.net/data_and_products/gridded_bathymetry_data/gebco_2023/.

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


## NOAA SFMR data

`noaa_met_data/20230829I1_A.nc`

`noaa_met_data/20230829I2_A.nc`

NOAA Stepped Frequency Microwave Radiometer data collected in Hurricane Idalia on 2023-08-29.
These data were originally accessed from https://www.aoml.noaa.gov/2023-hurricane-field-program-data/ on 28 October 2024.
These data are used in accordance with the NOAA/NWS Data and Products disclaimer at: https://www.weather.gov/disclaimer/.

This folder (and all of its files) should be downloaded and saved locally in this folder (`input_data/`) and the path should be stored in config.toml as `noaa_met_data`.
Available on the Dryad repository for this publication (https://doi.org/10.5061/dryad.zw3r228h7).

## USAFR SFMR data

`usafr_met_data/20230829U1.01.txt`

`usafr_met_data/20230830U1.01.txt`

`usafr_met_data/20230831U1.01.txt`

United States Air Force Reserve 53rd WRS Stepped Frequency Microwave Radiometer data collected in Hurricane Idalia 2023-08-29 to 2023-08-31.
These data were originally accessed from https://www.aoml.noaa.gov/2023-hurricane-field-program-data/ on 28 October 2024.
These data are used in accordance with the NOAA/NWS Data and Products disclaimer at: https://www.weather.gov/disclaimer/.

This folder (and all of its files) should be downloaded and saved locally in this folder (`input_data/`) and the path should be stored in config.toml as `usafr_met_data`.
Available on the Dryad repository for this publication (https://doi.org/10.5061/dryad.zw3r228h7).

## NDBC data

`NDBC_42036_202308_D7_v00.nc`

National Data Buoy Center Station 42036 (West Tampa) data for the month of 2023-08.  These data were originally accessed from https://www.ncei.noaa.gov/archive/accession/NDBC-CMANWx on 13 November 2023.
These data are used in accordance with the NOAA/NWS Data and Products disclaimer at: https://www.weather.gov/disclaimer/ and  "Use Limitations" under the "Description" tab at: https://www.ncei.noaa.gov/archive/accession/NDBC-CMANWx.

This file should be downloaded and saved locally in this folder (`input_data/`) and the path should be stored in config.toml as `ndbc_data`.
Available on the Dryad repository for this publication (https://doi.org/10.5061/dryad.zw3r228h7).
