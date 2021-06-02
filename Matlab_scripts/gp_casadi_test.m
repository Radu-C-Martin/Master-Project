clear all
close all
clc
%%%%%%%%%%%%
addpath('/home/radu/Media/MATLAB/casadi-linux-matlabR2014b-v3.5.5')
import casadi.*

%% Generate GP data
size = 500; n_samples = 15;
X = linspace(-2, 2, size);
Y = 3 * X .^2;

% Add noise to the output
mean = 0; std = 0.5;
noise = mean + std.*randn(1, n_samples);

idx_samples = randperm(size, n_samples);

X_sampled = X(idx_samples);
Y_sampled = Y(idx_samples);

Y_sampled = Y_sampled + noise;

figure; hold on; 
plot(X, Y);
scatter(X_sampled, Y_sampled);

tbl_gpr_in = array2table([X_sampled', Y_sampled']);
tbl_gpr_in.Properties.VariableNames = {'X', 'Y'};

tic;
model = fitrgp(tbl_gpr_in, 'Y', 'KernelFunction', 'ardsquaredexponential', ...
                'FitMethod', 'sr', 'PredictMethod', 'fic', 'Standardize', 1);
toc;

%% Predict stuff
[yhat_test, sigma_test] = predict(model, X');
std_test = sqrt(sigma_test);

% prepare it for the fill function
x_ax    = X';
X_plot  = [x_ax; flip(x_ax)];
Y_plot  = [yhat_test-1.96.*std_test; flip(yhat_test+1.96.*std_test)];

% plot a line + confidence bands
figure(); hold on;
title("GP performance on test data");
plot(x_ax, Y, 'red', 'LineWidth', 1.2);
plot(x_ax, yhat_test, 'blue', 'LineWidth', 1.2)
fill(X_plot, Y_plot , 1,....
        'facecolor','blue', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
legend({'data','prediction_mean', '95% confidence'},'Location','Best');
hold off

%% Save the model
save('test_gpr_model.mat', 'model')

%% CasADi optimization problem
cs_model = test_gpCallback('model');
cs_x = MX.sym('x');
cs_y = 2 * cs_model(cs_x) + 5;
f = Function('f', {cs_x}, {cs_y});


nlp_prob = struct('f', f(cs_x), 'x', cs_x);


opts = struct;
opts.ipopt.max_iter = 2000;
opts.ipopt.hessian_approximation = 'limited-memory';
%opts.ipopt.print_level =1;%0,3
opts.print_time = 0;
opts.ipopt.acceptable_tol =1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-6;

solver = nlpsol('solver', 'ipopt', nlp_prob,opts);

res = solver('lbx', -2, 'ubx', 2);

res

