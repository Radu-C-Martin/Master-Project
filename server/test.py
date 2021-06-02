from pathlib import Path
from shutil import copyfile
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.exceptions import NotFittedError

import gpflow
import tensorflow as tf

import matplotlib.pyplot as plt
from gpflow.utilities import print_summary

import casadi as cs

import callbacks
from helpers import *
from controllers import *

from time import sleep

t_cols = []
w_cols = ['SolRad', 'OutsideTemp']
u_cols = ['SimulatedHeat']
y_cols = ['SimulatedTemp']

t_lags = 0
w_lags = 1
u_lags = 2
y_lags = 3

dict_cols = {
    't': (t_lags, t_cols),
    'w': (w_lags, w_cols),
    'u': (u_lags, u_cols),
    'y': (y_lags, y_cols)
}

N_horizon = 10

controller = SVGP_MPCcontroller(dict_cols = dict_cols, N_horizon = N_horizon)



idx = 0
while True:
    u = controller.get_control_input()

    # Measure disturbance
    w = np.random.rand(2)
    controller.add_disturbance_measurement(w)
    w_forecast = np.random.rand(N_horizon, 2)
    controller.set_weather_forecast(w_forecast)

    # Measure output
    y = np.random.rand(1)
    controller.add_output_measurement(y)

    # Update model
    controller.update_model()
    
    idx += 1

pass
