import pickle
from pathlib import Path

import casadi as cs
import numpy as np
import pandas as pd

import gpflow
import tensorflow as tf

from gpflow.ci_utils import ci_niter

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import callbacks
from helpers import *


class PIDcontroller:
    """
    Generic PID controller
    """
    def __init__(self, P, I = 0, D = 0, arw_range = (-np.inf, np.inf)):
        self.P = P
        self.I = I
        self.D = D

        self.arw_range = arw_range

        self.ref = 25 # reference temperature
        self.y = 23 # T0

        self.err_acc = 0
        self.err_old = 0

    def add_output_measurement(self, y):
        self.y = y

    def get_control_input(self):

        err= self.ref - self.y
        self.err_acc += err

        # P
        sig_P = self.P * err

        # I
        sig_I = self.I * self.err_acc
        if sig_I < self.arw_range[0]:
            sig_I = self.arw_range[0]
        elif sig_I > self.arw_range[1]:
            sit_I = self.arw_range[1]

        # D
        sig_D = self.D * (err - self.err_old)

        self.err_old = err

        #print(f"P: {sig_P}, I: {sig_I}, D: {sig_D}")
        return sig_P + sig_I + sig_D

class sysIDcontroller(object):
    """
    Implementation of the PID controller with a disturbance signal, used for
    gathering experimental data before training a GP model
    """
    def __init__(self, u_range = (-1, 1)):
        self.u_range = u_range
        id_P = 30000
        id_I = 50000/(3 * 3600)
        id_D = 0
        self.PIDcontroller = PIDcontroller(P = id_P, I = id_I, D = id_D,
                arw_range = self.u_range)

    def get_control_input(self):

        # Input of PID controller
        sig_pid = self.PIDcontroller.get_control_input()
        # Random disturbance
        sig_w = (self.u_range[1] - self.u_range[0]) * np.random.rand() + self.u_range[0]
        # Combine and saturate
        print(f"sig_pid: {sig_pid}; sig_w: {sig_w}")
        sig = sig_pid + sig_w
        if sig < self.u_range[0]:
            sig = self.u_range[0]
        elif sig > self.u_range[1]:
            sig = self.u_range[1]

        return sig

    def add_output_measurement(self, y):
        self.PIDcontroller.add_output_measurement(y)

class Base_MPCcontroller(object):

    """
    Base Class for the MPC controller. Since the GP and SVGP controllers are
    mostly the same, with differences only in training/evaluation of the models,
    all the shared functions are implemented here, and the training/evaluation
    are implemented for the GP/SVGP models individually.
    """


    def __init__(self, dict_cols, model = None, scaler = None, N_horizon = 10, recover_from_crash = False):


        # Sample time (seconds)
        self.T_sample = 15 * 60 # Used for update frequency and reference
                                # calculation
        self.dict_cols = dict_cols
        self.max_lag = max([lag for lag,_ in self.dict_cols.values()])
        self.N_horizon = N_horizon
        self.n_states = np.sum([len(cols) * lags for lags,cols in
            self.dict_cols.values()])
        self.X_log = []

        # Adaptive models update parameters
        self.model_update_frequency = (24 * 3600)/self.T_sample # once per day
        self.model_update_frequency = 1000000
        self.pts_since_update = 0
        
        # Model log
        self.model_log = []

        ### Input range
        # Define an identification signal to be used first
        self.Pel = 2 * 6300 # [W]
        self.COP = 5.0

        # Set up identification controller
        self.u_range = self.COP * self.Pel * np.array([-1, 1])

        # Complete measurement history
        # Columns are: [SolRad, OutsideTemp] (Disturbance), Heat(Input), Temperature (Output)
        self.data_cols = []
        for _, cols in self.dict_cols.values():
            self.data_cols += cols
        self.data = np.empty((0, len(self.data_cols)))

        # Dataset used for training
        self.dataset_train_minsize = 5 * (24*3600)//self.T_sample # 5 days worth
        self.dataset_train_maxsize = np.iinfo(np.int32).max # maximum 32bit int
        #self.dataset_train_maxsize = 5 * (24*3600)//self.T_sample # 5 days worth
        self.dataset_train = np.empty((0, self.n_states))

        # The current weather forcast
        self.weather_forecast = None

        # Current measurements
        self.w, self.u, self.y = None, None, None

        # Solver parameters
        self.lbg = None
        self.ubg = None
        self.lbx = None
        self.ubx = None
        self.solver = None

        # Recover from a previous crash with precomputed values and continue
        self.recover_from_crash = recover_from_crash

        if self.recover_from_crash:
            self.model = pickle.load(open("controller_model.pkl", 'rb'))
            self.scaler = pickle.load(open("controller_scaler.pkl", 'rb'))
            self.scaler_helper = ScalerHelper(self.scaler)
            self.X_log = pickle.load(open("controller_X_log.pkl", 'rb'))
            df = pd.read_pickle("controller_df.pkl")
            self.recovery_signal = iter(df['SimulatedHeat'])
            return

        # Pre-existing model passed. Load all the necessary objects
        if model is not None:
            # Model is already trained. Using as is.
            self.id_mode = False
            if scaler is None:
                raise ValueError("Not allowed to pass a model without a scaler")
            self.model = model
            self.cs_model = callbacks.GPR("gpr", self.model, self.n_states)
            self.scaler = scaler
            self.scaler_helper = ScalerHelper(self.scaler)
        # No pre-existing model. Set up data acquisition and model training
        else:
            # No model has been passed. Setting up model initialization
            self.model = None
            self.id_mode = True

            self.id_controller = sysIDcontroller(self.u_range)

        return

    ###
    # Update measurements
    ###

    def add_disturbance_measurement(self, w):
        self.w = np.array(w).reshape(1, -1)

    def add_output_measurement(self, y):
        self.y = np.array(y).reshape(1, -1)
        # Also add measurement to ID controller if enabled
        if self.id_mode:
            print()
            self.id_controller.add_output_measurement(y)

    def _add_input_measurement(self, u):
        self.u = np.array(u).reshape(1, -1)

    def _add_measurement_set(self):
        new_data = np.hstack([self.w, self.u, self.y])
        self.data = np.vstack([self.data, new_data])
        print(f"{self.data.shape[0]} data points. Newest: {new_data}")
        self.w, self.u, self.y = None, None, None

    def set_weather_forecast(self, W):
        assert (W.shape[0] == self.N_horizon)
        self.weather_forecast = W

    ###
    # Set up optimal problem solver
    ###

    def _setup_solver(self):

        ###
        # Initialization
        ###
        self.cs_model = callbacks.GPR("gpr", self.model, self.n_states)

        X = cs.MX.sym("X", self.N_horizon + 1, self.n_states)
        x0 = cs.MX.sym("x0", 1, self.n_states)
        W = cs.MX.sym("W", self.N_horizon, 2)
        T_set_sc = cs.MX.sym("T_set_sc")

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
                    g.append(X[idx, base_col + w_base_col + w_lag_idx] - \
                            X[idx - 1, base_col + w_base_col + w_lag_idx - 1])
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

        p = cs.vertcat(cs.vec(W), cs.vec(x0), cs.vec(T_set_sc))

        prob = {'f': J, 'x': cs.vec(X), 'g': cs.vertcat(*g), 'p': p}
        options = {"ipopt": {"hessian_approximation": "limited-memory", "max_iter": 100,
                             "acceptable_tol": 1e-4, "tol": 1e-4,
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

    def _fit_scaler(self, nb_train_pts):
        # Get the min/max values from the dataset
        df = pd.DataFrame(self.data[-nb_train_pts:], columns = self.data_cols)

        # Get the range of data in the dataset
        df_range = df.describe().loc[['min', 'max'], :]
        
        # Manually overwrite the values for heat range
        in_names = self.dict_cols['u'][1]
        df_range.loc['min', in_names] = self.u_range[0]
        df_range.loc['max', in_names] = self.u_range[1]

        self.scaler = MinMaxScaler(feature_range = (-1, 1))
        self.scaler.fit(df_range.to_numpy())
        self.scaler_helper = ScalerHelper(self.scaler)
        return

    def _train_model(self):
        """
        Placeholder function to silence linter warning
        """
        raise NotImplementedError

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
                self._setup_solver()
                self.recover_from_crash = False

        # Training pre-loop

        if self.id_mode:
            if self.data.shape[0] < self.dataset_train_minsize:
                u = self.id_controller.get_control_input()
                self._add_input_measurement(u)
                return u
            else:
                # Collected enough data. Turn off identification mode
                self.id_mode = False
                # Training a model
                self._train_model()
                self._setup_solver()
                # Continue now since model exists

        # Model exists. Compute optimal control input


        data_scaled = self.scaler.transform(self.data)
        # Append a dummy row to data, to align data_to_gpr cols (which include measureemnt
        # at time t, with the current measurements, that stop at t-1)
        dummy_row = np.nan * np.ones((1, self.data.shape[1]))
        data_scaled = np.vstack([data_scaled, dummy_row])
        df = pd.DataFrame(data_scaled, columns = self.data_cols)

        x0 = data_to_gpr(df, self.dict_cols).drop(
                columns = self.dict_cols['w'][1] + self.dict_cols['u'][1] + self.dict_cols['y'][1]
                ).to_numpy()[-1, :]

        x0 = cs.vec(x0)

        # Compute mean outside temperature for last 48 hours
        nb_tref_points = 48 * 3600// self.T_sample # // for integer division
        T_out_avg = np.mean(self.data[-nb_tref_points:, 1])

        T_set = get_tref_mean(T_out_avg)
        T_set_sc = self.scaler_helper.scale_output(T_set)
        print(f"T_set: {T_set}  T_set_sc: {T_set_sc}")

        W = self.scaler_helper.scale_weather(self.weather_forecast)
        W = cs.vec(W)

        p = cs.vertcat(W, x0, T_set_sc)

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
        df_X = pd.DataFrame(X)
        u_idx = self.dict_cols['w'][0] * len(self.dict_cols['w'][1])
        # Take the first control action and apply it
        u = X[1, u_idx]
        # Unpack u after scaling since ScalerHelper returns a list/array
        [u] = self.scaler_helper.inverse_scale_input(u)
        self._add_input_measurement(u)
        return u

    def update_model(self):
        self._add_measurement_set()

        if not self.id_mode and not self.recover_from_crash:
            self.pts_since_update += 1

            if self.pts_since_update >= self.model_update_frequency:
                print(f"Updating model after {self.pts_since_update} measurements")
                # Append old model to log
                self.model_log.append(self.model)
                # Train new model
                self._train_model()
                # Re-initialize CasADi solver
                self._setup_solver()
                self.pts_since_update = 0

    def save_data(self):
        df = pd.DataFrame(self.data, columns = self.data_cols)
        df.to_pickle("controller_df.pkl")

        pickle.dump(self.scaler, open(Path("controller_scaler.pkl"), 'wb'))
        pickle.dump(self.model, open(Path("controller_model.pkl"), 'wb'))
        pickle.dump(self.X_log, open(Path("controller_X_log.pkl"), 'wb'))
        pickle.dump(self.model_log, open(Path("controller_model_log.pkl"), 'wb'))


class GP_MPCcontroller(Base_MPCcontroller):
    def __init__(self, dict_cols, model = None, scaler = None, N_horizon = 10, recover_from_crash = False):
        super().__init__(dict_cols, model, scaler, N_horizon, recover_from_crash)

    ###
    # GPflow model training and update
    ###
    def _train_model(self):
        """ Identify model from gathered data """

        nb_train_pts = self.dataset_train_minsize
        ###
        # Dataset
        ###
        # Train the model on the last nb_train_pts
        df = pd.DataFrame(self.data[-nb_train_pts:], columns = self.data_cols)
        self._fit_scaler(nb_train_pts)

        np_sc = self.scaler.transform(df.to_numpy())
        df_sc = pd.DataFrame(np_sc, index = df.index, columns = df.columns)
        df_gpr_train = data_to_gpr(df_sc, self.dict_cols)

        df_input_train = df_gpr_train.drop(columns = self.dict_cols['w'][1] + self.dict_cols['u'][1] + self.dict_cols['y'][1])
        df_output_train = df_gpr_train[self.dict_cols['y'][1]]

        np_input_train = df_input_train.to_numpy()
        np_output_train = df_output_train.to_numpy().reshape(-1, 1)

        data_train = (np_input_train, np_output_train)

        #df_test = pd.DataFrame(self.data[nb_train_pts:], columns = self.data_cols)
        #df_test_sc = get_scaled_df(df_test, self.dict_cols, self.scaler)
        #df_gpr_test = data_to_gpr(df_test_sc, self.dict_cols)
        #df_input_test = df_gpr_test.drop(columns = self.dict_cols['w'][1] + self.dict_cols['u'][1] + self.dict_cols['y'][1])
        #df_output_test = df_gpr_test[self.dict_cols['y'][1]]
        #np_input_test = df_input_test.to_numpy()
        #np_output_test = df_output_test.to_numpy()

        ###
        # Kernel
        ###

        nb_dims = np_input_train.shape[1]
        rational_dims = np.arange(0, (self.dict_cols['t'][0] + 1) * len(self.dict_cols['t'][1]), 1)
        nb_rational_dims = len(rational_dims)
        squared_dims = np.arange(nb_rational_dims, nb_dims, 1)
        nb_squared_dims = len(squared_dims)

        default_lscale = 1
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
                default_lscale += 10

        ###
        # Save model
        ###
        self.model = m

        return

class SVGP_MPCcontroller(Base_MPCcontroller):
    def __init__(self, dict_cols, model = None, scaler = None, N_horizon = 10, recover_from_crash = False):
        super().__init__(dict_cols, model, scaler, N_horizon, recover_from_crash)


    ###
    # GPflow model training and update
    ###
    def _train_model(self):
        """ Identify model from gathered data """

        nb_train_pts = self.dataset_train_maxsize
        ###
        # Dataset
        ###
        df = pd.DataFrame(self.data[-nb_train_pts:], columns = self.data_cols)
        print(f"Training model on {df.shape[0]} datapoints")
        self._fit_scaler(nb_train_pts)

        np_sc = self.scaler.transform(df.to_numpy())
        df_sc = pd.DataFrame(np_sc, index = df.index, columns = df.columns)
        df_gpr_train = data_to_gpr(df_sc, self.dict_cols)

        df_input_train = df_gpr_train.drop(columns = self.dict_cols['w'][1] + self.dict_cols['u'][1] + self.dict_cols['y'][1])
        df_output_train = df_gpr_train[self.dict_cols['y'][1]]

        np_input_train = df_input_train.to_numpy()
        np_output_train = df_output_train.to_numpy().reshape(-1, 1)

        data_train = (np_input_train, np_output_train)

        ###
        # Kernel
        ###

        nb_dims = np_input_train.shape[1]
        rational_dims = np.arange(0, (self.dict_cols['t'][0] + 1) * len(self.dict_cols['t'][1]), 1)
        nb_rational_dims = len(rational_dims)
        squared_dims = np.arange(nb_rational_dims, nb_dims, 1)
        nb_squared_dims = len(squared_dims)

        default_lscale = 1

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

        N = data_train[0].shape[0]
        M = 150 # Number of inducing locations
        Z = data_train[0][:M, :].copy()

        m = gpflow.models.SVGP(k, gpflow.likelihoods.Gaussian(), Z, num_data = N)

        elbo = tf.function(m.elbo)

        ###
        # Training
        ###

        minibatch_size = 100
        train_dataset = tf.data.Dataset.from_tensor_slices(data_train).repeat().shuffle(N)

        # Turn off training for inducing point locations
        gpflow.set_trainable(m.inducing_variable, False)

        def run_adam(model, iterations):
            """
            Utility function running the Adam optimizer

            :param model: GPflow model
            :param interations: number of iterations
            """
            # Create an Adam Optimizer action
            logf = []
            train_iter = iter(train_dataset.batch(minibatch_size))
            training_loss = model.training_loss_closure(train_iter, compile=True)
            optimizer = tf.optimizers.Adam()

            @tf.function
            def optimization_step():
                optimizer.minimize(training_loss, model.trainable_variables)

            for step in range(iterations):
                optimization_step()
                if step % 10 == 0:
                    elbo = -training_loss().numpy()
                    logf.append(elbo)
            return logf

        
        maxiter = ci_niter(10000)
        logf = run_adam(m, maxiter)

        ###
        # Save model
        ###
        self.model = m

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
