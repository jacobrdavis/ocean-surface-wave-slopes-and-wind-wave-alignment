# Code for: Ocean surface wave slopes and wind-wave alignment observed in Hurricane Idalia

## Abstract

Drifting buoy observations in Hurricane Idalia (2023) are used to investigate the dependence of ocean surface wave mean square slope on wind, wave, and storm characteristics.
Mean square slope has a primary dependence on wind speed that is linear at low-to-moderate wind speeds and approaches saturation at high wind speeds ($>$ 20 m/s).
Inside Hurricane Idalia, buoy-measured mean square slopes have a secondary dependence on wind-wave alignment:
at a given wind speed, slopes are higher where wind and waves are aligned compared to where wind and waves are crossing.
At moderate wind speeds, differences in mean square slope between aligned and crossing conditions can vary 10\% to 15\% relative to their mean.
The dependence on wind-wave alignment is robust up to 30 m/s, but can be obscured at the highest wind speeds near the center of the storm where wind and wave directions change rapidly. 
These changes in wave slopes may be related to the reported dependence of air-sea drag coefficients on wind-wave alignment.

## Data

All data are available in the Dryad repository for this publication (https://doi.org/10.5061/dryad.zw3r228h7) and in the Dryad repository for Davis et al. (2023) (https://doi.org/10.5061/dryad.g4f4qrfvb). 

Input data should be saved in [input_data/](input_data/), and paths are saved in [config.toml](config.toml).  See [input_data/README.md](input_data/README.md).  

## Structure

Analysis is organized into 11 Jupyter notebooks (.ipynb).  Notebooks are named in the order they should be run (e.g. nb0, nb1, ..., nb10).  Variables are shared between notebooks, and notebooks call prior notebooks as needed.

Most functions are organized into modules within [src/](src/).  Much of the code is from a standalone package, but it is copied here for archival purposes.

## Installation

1. Clone this repository.
2. Download the data and move it to [input_data/](input_data/). (See the **Data** section above.)
3. Create a Python environment.  If using conda, run:
   ```sh
   conda env create -f environment.yml
   ```
4. Run any of the .ipynb notebooks.
