import pickle
from pathlib import Path

import casadi as cs
import numpy as np
import pandas as pd

import gpflow
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

import callbacks
from helpers import *


class PIDcontroller:
    def __init__(self, P, I = 0, D = 0):
        self.P = P
        self.I = I
        self.D = D

        self.ref = 22

        self.err_acc = 0
        self.err_old = 0

    def get_control(self, y):
        """
        y: Measured output
        """

        err= self.ref - y
        self.err_acc += err

        sig_P = self.P * err
        sig_I = self.I * self.err_acc
        sig_D = self.D * (err - self.err_old)

        self.err_old = err

        #print(f"P: {sig_P}, I: {sig_I}, D: {sig_D}")
        return sig_P + sig_I + sig_D



class GP_MPCcontroller:
    def __init__(self, dict_cols, model = None, scaler = None, N_horizon = 10, recover_from_crash = False):


        self.recover_from_crash = recover_from_crash

        if self.recover_from_crash:
            self.model = pickle.load(open("controller_model.pkl", 'rb'))
            self.scaler = pickle.load(open("controller_scaler.pkl", 'rb'))
            self.X_log = pickle.load(open("controller_X_log.pkl", 'rb'))
            df = pd.read_pickle("controller_df.pkl")
            self.recovery_signal = iter(df['SimulatedHeat'])


        if model is not None:
            # Model is already trained. Using as is.
            if scaler is None: raise ValueError("Not allowed to pass a model without a scaler")
            self.model = model
            self.cs_model = callbacks.GPR("gpr", self.model)
            self.scaler = scaler
            self.scaler_helper = ScalerHelper(self.scaler)
        else:
            # No model has been passed. Setting up model initialization
            self.model = None
            self.nb_data = 500 + 1
            self.ident_signal = get_random_signal(self.nb_data, signal_type = 'analog')


            self.Pel = 2 * 6300
            self.COP = 5.0

            self.ident_signal = iter(self.Pel * self.COP * self.ident_signal)


        self.dict_cols = dict_cols
        self.max_lag = max([lag for lag,_ in self.dict_cols.values()])
        self.N_horizon = N_horizon
        self.X_log = []


        # Complete measurement history
        # Columns are: [SolRad, OutsideTemp] (Disturbance), Heat(Input), Temperature (Output)
        self.data_cols = []
        for lags, cols in self.dict_cols.values():
            self.data_cols += cols
        self.data = np.empty((0, len(self.data_cols)))

        # The current weather forcast
        self.weather_forecast = None

        # Current measurements
        self.w, self.u, self.y = None, None, None



    ###
    # GPflow model training and update
    ###
    def _train_model(self):
        """ Identify model from gathered data """

        nb_train_pts = self.nb_data - 1
        ###
        # Dataset
        ###
        df = pd.DataFrame(self.data[:nb_train_pts], columns = self.data_cols)
        self.scaler = MinMaxScaler(feature_range = (-1, 1)) 
        self.scaler_helper = ScalerHelper(self.scaler)
        df_sc = get_scaled_df(df, self.dict_cols, self.scaler)
        df_gpr_train = data_to_gpr(df_sc, self.dict_cols)

        df_input_train = df_gpr_train.drop(columns = self.dict_cols['w'][1] + self.dict_cols['u'][1] + self.dict_cols['y'][1])
        df_output_train = df_gpr_train[self.dict_cols['y'][1]]

        np_input_train = df_input_train.to_numpy()
        np_output_train = df_output_train.to_numpy().reshape(-1, 1)

        data_train = (np_input_train, np_output_train)

        df_test = pd.DataFrame(self.data[nb_train_pts:], columns = self.data_cols)
        df_test_sc = get_scaled_df(df_test, self.dict_cols, self.scaler)
        df_gpr_test = data_to_gpr(df_test_sc, self.dict_cols)
        df_input_test = df_gpr_test.drop(columns = self.dict_cols['w'][1] + self.dict_cols['u'][1] + self.dict_cols['y'][1])
        df_output_test = df_gpr_test[self.dict_cols['y'][1]]
        np_input_test = df_input_test.to_numpy()
        np_output_test = df_output_test.to_numpy()

        ###
        # Kernel
        ###

        nb_dims = np_input_train.shape[1]
        rational_dims = np.arange(0, (self.dict_cols['t'][0] + 1) * len(self.dict_cols['t'][1]), 1)
        nb_rational_dims = len(rational_dims)
        squared_dims = np.arange(nb_rational_dims, nb_dims, 1)
        nb_squared_dims = len(squared_dims)

        default_lscale = 75
        while True:
            try:

                squared_l = np.linspace(default_lscale, 2*default_lscale, nb_squared_dims)
                rational_l = np.linspace(default_lscale, 2*default_lscale, nb_rational_dims)

                variance = tf.math.reduce_variance(np_input_train)

                k0 = gpflow.kernels.SquaredExponential(lengthscales = squared_l, active_dims = squared_dims, variance = variance)
                k1 = gpflow.kernels.Constant(variance = variance)
                k2 = gpflow.kernels.RationalQuadratic(lengthscales = rational_l, active_dims = rational_dims, variance = variance)
                k3 = gpflow.kernels.Periodic(k2)

                k = (k0 + k1) * k2
                k = k0

                ###
                # Model
                ###

                m = gpflow.models.GPR(
                    data = data_train,
                    kernel = k,
                    mean_function = None,
                )

                ###
                # Training
                ###
                print(f"Training a model with lscale:{default_lscale}")
                opt = gpflow.optimizers.Scipy()
                opt.minimize(m.training_loss, m.trainable_variables)
                break
            except:
                print(f"Failed.Increasing lengthscale")
                default_lscale += 5

        ###
        # Save model
        ###
        self.model = m
        self.n_states = self.model.data[0].shape[1]

#        # Manual model validation
#        import matplotlib.pyplot as plt
#
#        plt.figure()
#        plt.plot(data_train[1], label = 'real')
#        mean, var = self.model.predict_f(data_train[0])
#        plt.plot(mean, label = 'model')
#        plt.legend()
#        plt.show()
#
#        plt.figure()
#        plt.plot(np_output_test, label = 'real')
#        mean, var = self.model.predict_f(np_input_test)
#        plt.plot(mean, label = 'model')
#        plt.legend()
#        plt.show()
#
#        import pdb; pdb.set_trace()
        pass

    ###
    # Update measurements
    ###

    def add_disturbance_measurement(self, w):
        self.w = np.array(w).reshape(1, -1)

    def add_output_measurement(self, y):
        self.y = np.array(y).reshape(1, -1)

    def _add_input_measurement(self, u):
        self.u = np.array(u).reshape(1, -1)

    def set_weather_forecast(self, W):
        assert (W.shape[0] == self.N_horizon)
        self.weather_forecast = W

    def update_model(self):
        new_data = np.hstack([self.w, self.u, self.y])
        print(f"Adding new data: {new_data}")
        self.data = np.vstack([self.data, new_data])
        print(f"Data size: {self.data.shape[0]}")
        self.w, self.u, self.y = None, None, None
        print("---")

    ###
    # Set up optimal problem solver
    ###

    def _setup_solver(self):

        ###
        # Initialization
        ###
        self.cs_model = callbacks.GPR("gpr", self.model)

        T_set = 21
        T_set_sc = self.scaler_helper.scale_output(T_set)


        X = cs.MX.sym("X", self.N_horizon + 1, self.n_states)
        x0 = cs.MX.sym("x0", 1, self.n_states)
        W = cs.MX.sym("W", self.N_horizon, 2)

        g = []
        lbg = []
        ubg = []

        lbx = -np.inf*np.ones(X.shape)
        ubx = np.inf*np.ones(X.shape)

        u_idx = self.dict_cols['w'][0] * len(self.dict_cols['w'][1])
        y_idx = u_idx + self.dict_cols['u'][0] * len(self.dict_cols['u'][1])
        # Stage cost
        u_cost = 0.05 * cs.dot(X[:, u_idx], X[:, u_idx])
        u_cost = 0
        y_cost = cs.dot(X[:, y_idx] - T_set_sc, X[:, y_idx] - T_set_sc)
        J = u_cost + y_cost

        # Set up equality constraints for the first step
        for idx in range(self.n_states):
            g.append(X[0, idx] - x0[0, idx])
            lbg.append(0)
            ubg.append(0)

        # Set up equality constraints for the following steps
        for idx in range(1, self.N_horizon + 1):
            base_col = 0
            # w
            nb_cols = self.dict_cols['w'][0]
            for w_idx in range(W.shape[1]):
                w_base_col = w_idx * nb_cols
                g.append(X[idx, base_col + w_base_col] - W[idx - 1, w_idx])
                lbg.append(0)
                ubg.append(0)
                for w_lag_idx in range(1, nb_cols):
                    g.append(X[idx, base_col + w_base_col + w_lag_idx] - X[idx - 1, base_col + w_base_col + w_lag_idx - 1])
                    lbg.append(0)
                    ubg.append(0)
                    
            base_col += nb_cols * W.shape[1]
            # u
            nb_cols = self.dict_cols['u'][0]

            lbx[idx, base_col] = -1 #lower bound on input
            ubx[idx, base_col] = 1 #upper bound on input
            for u_lag_idx in range(1, nb_cols):
                g.append(X[idx, base_col + u_lag_idx] - X[idx - 1, base_col + u_lag_idx - 1])
                lbg.append(0)
                ubg.append(0)
            
            base_col += nb_cols
            # y
            nb_cols = self.dict_cols['y'][0]
            g.append(X[idx, base_col] - self.cs_model(X[idx - 1, :]))
            lbg.append(0)
            ubg.append(0)
            for y_lag_idx in range(1, nb_cols):
                g.append(X[idx, base_col + y_lag_idx] - X[idx - 1, base_col + y_lag_idx - 1])
                lbg.append(0)
                ubg.append(0)

        p = cs.vertcat(cs.vec(W), cs.vec(x0))

        prob = {'f': J, 'x': cs.vec(X), 'g': cs.vertcat(*g), 'p': p}
        options = {"ipopt": {"hessian_approximation": "limited-memory", "max_iter": 100,
                             #"acceptable_tol": 1e-6, "tol": 1e-6,
                             #"linear_solver": "ma57",
                             #"acceptable_obj_change_tol": 1e-5,
                             #"mu_strategy": "adaptive",
                             "print_level": 0
                            }}

        self.lbg = lbg
        self.ubg = ubg
        self.lbx = lbx
        self.ubx = ubx
        self.solver = cs.nlpsol("solver","ipopt",prob, options)

        print("Initialized casadi solver")

    ###
    # Compute control signal
    ###

    def get_control_input(self):

        # Recovery pre-loop
        if self.recover_from_crash:
            try:
                u = next(self.recovery_signal)
                self._add_input_measurement(u)
                return u
            except StopIteration:
                # Finished passing in the pre-existing control
                # Switching to normal operation
                self.recover_from_crash = False

        # Training pre-loop
        if self.model is None:
            try:
                # No model yet. Sending next step of identification signal
                u = next(self.ident_signal)
                self._add_input_measurement(u)
                return u
            except StopIteration:
                # No more identification signal. Training a model and proceeding
                self._train_model()
                self._setup_solver()
                # Continue now since model exists

        # Model exists. Compute optimal control input


        data_scaled = self.scaler.transform(self.data)
        df = pd.DataFrame(data_scaled, columns = self.data_cols)
        
        x0 = data_to_gpr(df, self.dict_cols).drop(
                columns = self.dict_cols['w'][1] + self.dict_cols['u'][1] + self.dict_cols['y'][1]
                ).to_numpy()[-1, :]

        x0 = cs.vec(x0)

        W = self.scaler_helper.scale_weather(self.weather_forecast)
        W = cs.vec(W)
        
        p = cs.vertcat(W, x0)

        res = self.solver(
                x0 = 1,
                lbx = cs.vec(self.lbx),
                ubx = cs.vec(self.ubx),
                p = p,
                lbg = cs.vec(self.lbg),
                ubg = cs.vec(self.ubg)
                )

        X = np.array(res['x'].reshape((self.N_horizon + 1, self.n_states)))
        self.X_log.append(X)
        u_idx = self.dict_cols['w'][0] * len(self.dict_cols['w'][1])
        # Take the first control action and apply it
        u = X[1, u_idx]
        # Unpack u after scaling since ScalerHelper returns a list/array
        [u] = self.scaler_helper.inverse_scale_input(u)
        self._add_input_measurement(u)
        return u

    def save_data(self):
        df = pd.DataFrame(self.data, columns = self.data_cols)
        df.to_pickle("controller_df.pkl")

        pickle.dump(self.scaler, open(Path("controller_scaler.pkl"), 'wb'))
        pickle.dump(self.model, open(Path("controller_model.pkl"), 'wb'))
        pickle.dump(self.X_log, open(Path("controller_X_log.pkl"), 'wb'))
        
        return


class TestController:
    def __init__(self):
        return

    def get_control_input(self): return 2 * 3600 * 5.0 * np.random.rand()
    def add_disturbance_measurement(self, w): return
    def set_weather_forecast(self, W): return
    def add_output_measurement(self, y): return
    def update_model(self): return
    def save_data(self): return
