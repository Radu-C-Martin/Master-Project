clear all
close all
clc

%% Import CasADi
addpath('/home/radu/Media/MATLAB/casadi-linux-matlabR2014b-v3.5.5')
import casadi.*

load("gpr_model.mat")
%% Initialize casadi callback
cs_model = gpCallback('model');

T_set = 20;
N_horizon = 5;
n_states = 7;


COP = 5; %cooling
EER = 5; %heating
Pel = 6300; % Electric Power Consumption of the HVAC

u_min = - COP * Pel;
u_max =   EER * Pel;

J = 0; % optimization objective
g = []; % constraints vector

% Set up the symbolic variables
U = MX.sym('U', N_horizon, 1);
W = MX.sym('W', N_horizon, 2);
x0 = MX.sym('x0', 1, n_states - 3);

% setup the first state
wk = W(1, :);
uk = U(1); % scaled input
xk = [wk, Pel*uk, x0];
yk = cs_model(xk);
J = J + (yk - T_set).^2;

% Setup the rest of the states
for idx = 2:N_horizon
    wk = W(idx, :); 
    uk_1 = uk; uk = U(idx);
    xk = [wk, Pel*uk, Pel*uk_1, yk, xk(5:6)];
    yk = cs_model(xk);
    J = J + (yk - T_set).^2;
end
    
p = [vec(W); vec(x0)];

nlp_prob = struct('f', J, 'x', vec(U), 'g', g, 'p', p);

opts = struct;
%opts.ipopt.max_iter = 5000;
opts.ipopt.max_cpu_time = 15*60;
opts.ipopt.hessian_approximation = 'limited-memory';
%opts.ipopt.print_level =1;%0,3
opts.print_time = 0;
opts.ipopt.acceptable_tol =1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-6;

solver = nlpsol('solver', 'ipopt', nlp_prob,opts);

real_x0 = [0,  23,  23,  23];
real_W = [[57.9261000000000;54.9020333333334;73.8607000000000;76.0425333333333;64.9819666666667], [22; 22; 22; 22; 22]];

real_p = vertcat(vec(DM(real_W)), vec(DM(real_x0)));


res = solver('p', real_p, 'ubx', EER, 'lbx', -COP);
%% Interpret the optimization result
x = Pel * full(res.x);
X = [real_W, x, [real_x0; zeros(N_horizon -1, size(real_x0, 2))]];

X(2:end, 4) = X(1:end-1, 3);

for idx=2:N_horizon
    X(idx, 5) = full(cs_model(X(idx - 1, :)));
    X(idx, 6:7) = X(idx - 1, 5:6);
end
T_horizon = cs_model(X');


figure; hold on;
plot(1:N_horizon, full(T_horizon));
plot(1:N_horizon, T_set*ones(1, N_horizon));