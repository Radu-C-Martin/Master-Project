import pandas as pd
import numpy as np

import gpflow
import tensorflow as tf

from sklearn.exceptions import NotFittedError

# Generator for tensorflow functions for a given model
def get_model_evaluator(model):
    @tf.function
    def model_evaluator(tf_input):
        preds = model.predict_f(tf_input)
        return preds
    return model_evaluator

def get_grad_evaluator(model):
    @tf.function
    def grad_evaluator(tf_input):
        with tf.GradientTape() as tape:
            preds = model.predict_f(tf_input)
        grads = tape.gradient(preds, tf_input)
        return grads
    return grad_evaluator

def get_combined_evaluator(model):
    @tf.function
    def combined_evaluator(tf_input):
        with tf.GradientTape() as tape:
            preds = model.predict_f(tf_input)
        grads = tape.gradient(preds, tf_input)
        return preds, grads
    return combined_evaluator

def get_random_signal(nstep, a_range = (-1, 1), b_range = (2, 10), signal_type = 'analog'):

    a = np.random.rand(nstep) * (a_range[1]-a_range[0]) + a_range[0] # range for amplitude
    b = np.random.rand(nstep) *(b_range[1]-b_range[0]) + b_range[0] # range for frequency
    b = np.round(b)
    b = b.astype(int)

    b[0] = 0

    for i in range(1,np.size(b)):
        b[i] = b[i-1]+b[i]

    if signal_type == 'analog':
        # Random Signal
        i=0
        random_signal = np.zeros(nstep)
        while b[i]<np.size(random_signal):
            k = b[i]
            random_signal[k:] = a[i]
            i=i+1
        return random_signal
    elif signal_type == 'prbs':
        # PRBS
        a = np.zeros(nstep)
        j = 0
        while j < nstep - 1:
            a[j] = a_range[1]
            a[j+1] = a_range[0]
            j = j+2

        i=0
        prbs = np.zeros(nstep)
        while b[i]<np.size(prbs):
            k = b[i]
            prbs[k:] = a[i]
            i=i+1

        return prbs
    else:
        raise ValueError(signal_type)


def get_identification_signal(size):
    # Base random signal
    rand_signal = get_random_signal(size, signal_type = 'prbs')
    # Integrator (cumulative sum)
    cum_signal = 3/size * np.ones((1, size))
    cum_signal = np.cumsum(cum_signal)
    # Combine signals and clip signal to [-1, 1] range
    ident_signal = rand_signal + cum_signal
    ident_signal = np.where(ident_signal < -1, -1, ident_signal)
    ident_signal = np.where(ident_signal > 1, 1, ident_signal)

    return ident_signal


def get_scaled_df(df, dict_cols, scaler):

    """
    Scale the dataframe with the given scaler. Drops unused columns.
    """

    t_list = dict_cols['t'][1]
    w_list = dict_cols['w'][1]
    u_list = dict_cols['u'][1]
    y_list = dict_cols['y'][1]

    df_local = df[t_list + w_list + u_list + y_list]
    df_scaled = df_local.to_numpy()

    try:
        df_scaled = scaler.transform(df_scaled)
    except NotFittedError:
        df_scaled = scaler.fit_transform(df_scaled)

    df_scaled = pd.DataFrame(
            df_scaled,
            index = df_local.index,
            columns = df_local.columns
    )

    return df_scaled

def data_to_gpr(df, dict_cols):

    t_list = dict_cols['t'][1]
    w_list = dict_cols['w'][1]
    u_list = dict_cols['u'][1]
    y_list = dict_cols['y'][1]

    df_gpr = df[t_list + w_list + u_list + y_list].copy()

    for lags, names in dict_cols.values():
        for name in names:
            col_idx = df_gpr.columns.get_loc(name)
            for lag in range(1, lags + 1):
                df_gpr.insert(
                        col_idx + lag,
                        f"{name}_{lag}",
                        df_gpr.loc[:, name].shift(lag)
                )
    # Drop empty rows (first rows without enough lags)
    df_gpr.dropna(inplace = True)

    return df_gpr

def get_gp_model(data, dict_cols):
   
    nb_dims = data[0].shape[1]
    rational_dims = np.arange(0, (dict_cols['t'][0] + 1) * len(dict_cols['t'][1]), 1)
    nb_rational_dims = len(rational_dims)
    squared_dims = np.arange(nb_rational_dims, nb_dims, 1)
    nb_squared_dims = len(squared_dims)

    squared_l = np.linspace(10, 10, nb_squared_dims)
    rational_l = np.linspace(10, 10, nb_rational_dims)

    variance = 1
    # Define the kernel to be used
    k0 = gpflow.kernels.SquaredExponential(
            lengthscales = squared_l,
            active_dims = squared_dims,
            variance = variance
    )
    k1 = gpflow.kernels.Constant(
            variance = variance
    )
    k2 = gpflow.kernels.RationalQuadratic(
            lengthscales = rational_l,
            active_dims = rational_dims,
            variance = variance
    )
    #k3 = gpflow.kernels.Periodic(k2)
    k4 = gpflow.kernels.Linear(variance = [1]*nb_dims)

    k = k4

    model = gpflow.models.GPR(
                data = data,
                kernel = k,
                mean_function = None
    )

    return model

class ScalerHelper:

    """
    Column order in scaler should be as follows:
    t_cols + w_cols + u_cols + y_cols
    """

    def __init__(self, scaler):
        self.scaler = scaler

    def scale_time(self, t):
        raise NotImplementedError

    def inverse_scale_time(self, t):
        raise NotImplementedError

    def scale_weather(self, W):
        W_local = np.array(W).reshape(-1, 2)
        W_filled = np.hstack([W_local, np.zeros((W_local.shape[0], 4 - W_local.shape[1]))]) 
        W_scaled = self.scaler.transform(W_filled)
        W_scaled = W_scaled[:, :2]
        return W_scaled

    def inverse_scale_weather(self, W):
        W_local = np.array(W).reshape(-1, 2)
        W_filled = np.hstack([W_local, np.zeros((W_local.shape[0], 4 - W_local.shape[1]))]) 
        W_scaled = self.scaler.inverse_transform(W_filled)
        W_scaled = W_scaled[:, :2]
        return W_scaled


    def scale_input(self, U):
        U_local = np.array(U).reshape(-1, 1)
        U_filled = np.hstack([np.zeros((U_local.shape[0], 2)), U_local, np.zeros((U_local.shape[0], 1))])
        U_scaled = self.scaler.transform(U_filled)
        U_scaled = U_scaled[:, 2]
        return U_scaled

    def inverse_scale_input(self, U):
        U_local = np.array(U).reshape(-1, 1)
        U_filled = np.hstack([np.zeros((U_local.shape[0], 2)), U_local, np.zeros((U_local.shape[0], 1))])
        U_scaled = self.scaler.inverse_transform(U_filled)
        U_scaled = U_scaled[:, 2]
        return U_scaled

    def scale_output(self, Y):
        Y_local = np.array(Y).reshape(-1, 1)
        Y_filled = np.hstack([np.zeros((Y_local.shape[0], 3)), Y_local])
        Y_scaled = self.scaler.transform(Y_filled)
        Y_scaled = Y_scaled[:, 3]
        return Y_scaled

    def inverse_scale_output(self, Y):
        Y_local = np.array(Y).reshape(-1, 1)
        Y_filled = np.hstack([np.zeros((Y_local.shape[0], 3)), Y_local])
        Y_scaled = self.scaler.inverse_transform(Y_filled)
        Y_scaled = Y_scaled[:, 3]
        return Y_scaled
