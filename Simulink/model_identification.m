clear all
close all
clc
%%%%%%%%%%%%%%%%%%%%%%%

%% Set the run parameters

% Set the initial temperature to be the measured initial temperature
t0 = 23;

runtime1 = 161400;
runtime2 = 136200;
runtime3 = 208200;
runtime4 = 208200;
runtime5 = 208200;
runtime6 = 208200;
runtime7 = 553800;

runtime = 24 * 3600;
set_param('polydome', 'StopTime', int2str(runtime))
Tsample = 900; 
steps = runtime/Tsample;
tin = Tsample *(0:steps)';

prbs_sig = 2*prbs(8, steps+1)' - 1;
COP = 5.0;
Pel = 6300;


power = [tin COP*Pel*prbs_sig(1:steps+1)];

%% Simulate the model
out = sim('polydome');

%% For manual simulation running
WeatherMeasurement = struct;
WeatherMeasurement.data = squeeze(out.WeatherMeasurement.data)';
WeatherMeasurement.time = out.WeatherMeasurement.time;

input = [power(:, 2:end) WeatherMeasurement.data];

Exp7_data = iddata(out.SimulatedTemp.data, input);

Exp7_table = array2table([input out.SimulatedTemp.data], 'VariableNames', {'Power', 'SolRad', 'OutsideTemp', 'SimulatedTemp'});

writetable(Exp7_table, 'Exp7_table.csv')

%%
save('Exp_CARNOT.mat', ...
    'Exp1_data', 'Exp1_table', ...
    'Exp2_data', 'Exp2_table', ...
    'Exp3_data', 'Exp3_table', ...
    'Exp4_data', 'Exp4_table', ...
    'Exp5_data', 'Exp5_table', ...
    'Exp6_data', 'Exp6_table', ...
    'Exp7_data', 'Exp7_table'  ...
)

data_train = merge(Exp1_data, Exp3_data, Exp5_data);
data_test = merge(Exp2_data, Exp4_data, Exp6_data, Exp7_data);
