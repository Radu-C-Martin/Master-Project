# Inter-seasonal GP MPC control for buildings

## Identification and control code

This repository contains the code of the Notebooks and the server implementation
of the GP-based MPC used as part of my Master Project 

> *Inter-seasonal Performance of Gaussian Process-based 
> Model Predictive Control of Buildings*

which was done under the supervision of Prof. Colin Jones at the EPFL's
[Automatic Control Laboratory](https://www.epfl.ch/labs/la/)

## Folder structure

The most important parts of the repository are presented in the tree below:

```shell
Master-Project
├── Data
│   ├── Experimental_data_WDB
│   ├── Good_CARNOT
│   ├── Luca_experimental_data
│   ├── Miscellaneous
│   └── Simulation_results
│       ├── First batch (explorative)
│       ├── Second batch
│       └── Third batch
│           ├── 10_SVGP_480pts_inf_window_12_averageYear_LinearKernel
│           ├── 1_SVGP_480pts_inf_window_12_averageYear
│           ├── 2_SVGP_480pts_inf_window_12_extremeSummer
│           ├── 3_SVGP_480pts_inf_window_12_extremeWinter
│           ├── 4_GP_480pts_12_averageYear
│           ├── 5_SVGP_480pts_480pts_window_12_averageYear
│           └── 6_SVGP_96pts_inf_window_12_averageYear
├── Literature
├── Matlab_scripts
├── Notebooks
│   ├── Images
│   └── Results
├── server
└── Simulink
```

### `Data` folder

The `Data` folder contains all the experimental datasets
(`Luca_experimental_data`), the CARNOT simulations using the experimental
weather data (`Experimental_data_WDB`, used for validation of the CARNOT
building), as well as the models and optimization results for the full-year
simulations (in the sub-folder `Simulation_results/Third batch`)

### `Notebooks` folder

The `Notebooks` folder contains all the relevant jupyter notebooks written while
figuring out the structure of the MPC controller, the CARNOT system and their
interface. These notebooks are also used to generate most of the plots used in
the [Master Project Report](https://github.com/Radu-C-Martin/Master-Thesis).

A quick summary of each notebook's uses is presented in the following list:

- `10_wdb_from_experimental_data.ipynb`
    - Transforming experimental weather data to WDB

- `21_CARNOT_experimental_comparison.ipynb`
    - Graphing external temp/ internal temp/ setpoint for all exps

- `30_gaussiandome_identification.ipynb`
    - Gaussian Process identification for real building based on exp. data

- `38_gp_hyperparameter_estimation.ipynb`
    - Gaussian Process identification on CARNOT experimental data
    - Error table generation
    - Training/test error plots
    - Multi-step ahead prediction + plots

- `39_svgp_hyperparameter_estimation.ipynb`
    - SVGP model identification on CARNOT data
    - Error table generation
    - Training/test error plots
    - Multi-step ahead prediction + plots

- `42_casadi_callback_speed.ipynb`
    - CasADi GP with callback on a GP of the CARNOT building

- `50_mpc_formulation.ipynb`
    - Test of sample MPC with CasADi on a CARNOT GP

- `70_Server_result_analysis.ipynb`
    - Plots for analyzing year-long simulation performance

Experiments:

- `31_gpflow_first_test.ipynb`
    - Test of GPflow on a simple function

- `41_casadi_gp_test.ipynb`
    - CasADi GP with callback on a simple function

### `server` folder

The `server` folder contains the code for the Python server serving as the
interface between the Simulink model and the MPC controller. The `server.py`
script is responsible of reading the measurement values/weather forecast and
sending the control signal to the CARNOT model. The measurement and control
values are passed to/from and MPC object. The MPC itself is defined in
`controllers.py`.

## Required libraries

The python code uses several libraries for GP models, as well as general
math/data operations:

- `pandas` and `numpy` are used for operating on arrays of data
- `TensorFlow` and `GPflow` are used for the GP model implementations and
    to wrap the CasADi callbacks as tf-functions.
- `CasADi` is the algorithmic differentation framework used to implement
    the Optimization Problem

There are also several libraries that are optional, and are not required for
running the server code:

- `matplotlib` is used in the jupyter notebooks to plot the results

- `bokeh` is used in the jupyter notebooks where interactive plots were
    useful
- `tqdm` is used to display a progress bar in the notebooks where a lot of
    data is processed.
