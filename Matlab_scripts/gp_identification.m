clear all
close all
clc
%%%%%%%%%%%

load("gpr_carnot.mat");

%% Format the train/test data arrays
tbl_gpr_train = array2table(gpr_train);
tbl_gpr_train.Properties.VariableNames = cellstr(table_cols);
tbl_gpr_train = removevars(tbl_gpr_train,{'u'});
tbl_gpr_train_x = removevars(tbl_gpr_train, {'y'});

tbl_gpr_test = array2table(gpr_test);
tbl_gpr_test.Properties.VariableNames = cellstr(table_cols);
tbl_gpr_test = removevars(tbl_gpr_test,{'u'});
tbl_gpr_test_x = removevars(tbl_gpr_test, {'y'});



%% Train the GP model
OutputName = 'y';

tic;
model = fitrgp(tbl_gpr_train, OutputName, 'KernelFunction', 'ardsquaredexponential', ...
                'FitMethod', 'sr', 'PredictMethod', 'fic', 'Standardize', 1);
toc;
%% Validate the model using training data
[yhat_train, sigma_train] = predict(model, tbl_gpr_train_x);
std_train = sqrt(sigma_train);

% prepare it for the fill function
x_ax    = (1:size(tbl_gpr_train, 1))';
X_plot  = [x_ax; flip(x_ax)];
Y_plot  = [yhat_train-1.96.*std_train; flip(yhat_train+1.96.*std_train)];

% plot a line + confidence bands
figure(); hold on;
title("GP performance on training data");
plot(x_ax, tbl_gpr_train.y, 'red', 'LineWidth', 1.2);
plot(x_ax, yhat_train, 'blue', 'LineWidth', 1.2)
fill(X_plot, Y_plot , 1,....
        'facecolor','blue', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
legend({'data','prediction_mean', '95% confidence'},'Location','Best');
hold off 

%% Validate the model using test data
[yhat_test, sigma_test] = predict(model, tbl_gpr_test_x);
std_test = sqrt(sigma_test);

% prepare it for the fill function
x_ax    = (1:size(tbl_gpr_test, 1))';
X_plot  = [x_ax; flip(x_ax)];
Y_plot  = [yhat_test-1.96.*std_test; flip(yhat_test+1.96.*std_test)];

% plot a line + confidence bands
figure(); hold on;
title("GP performance on test data");
plot(x_ax, tbl_gpr_test.y, 'red', 'LineWidth', 1.2);
plot(x_ax, yhat_test, 'blue', 'LineWidth', 1.2)
fill(X_plot, Y_plot , 1,....
        'facecolor','blue', ...
        'edgecolor','none', ...
        'facealpha', 0.3);
legend({'data','prediction_mean', '95% confidence'},'Location','Best');
hold off 

%% Export the final GP model
save('gpr_model.mat', 'model')