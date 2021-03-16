clear all
close all
clc
%%%%%%%%%%%%%%%%%%%%%%%

%% Load the experimental data
exp_id = "Exp1";
exp_path = strcat("../Data/Luca_experimental_data/", exp_id,".mat");
wdb_path = strcat("../Data/Experimental_data_WDB/", exp_id, "_WDB.mat");

Exp_data = load(exp_path);
load(wdb_path);

% Save the current WDB to the Simulink model import (since Carnot's input file is hardcoded)
save('../Data/input_WDB.mat', 'Exp_WDB');

tin = Exp_WDB(:,1);

% The power trick: when the setpoint is larger than the actual temperature
% the HVAC system is heating the room, otherwise it is cooling the room
Setpoint = Exp_data.(exp_id).Setpoint.values;
InsideTemp = mean([Exp_data.(exp_id).InsideTemp.values, Exp_data.(exp_id).LakeTemp.values], 2);
OutsideTemp = Exp_data.(exp_id).OutsideTemp.values;

HVAC_COP = 3;
Heating_coeff = sign(Setpoint - InsideTemp);
Heating_coeff(Heating_coeff == -1) = -1 * HVAC_COP;


%% Set the model parameters

surface_part = 0.1;

%% Set the run parameters

air_exchange_rate = tin;
air_exchange_rate(:,2) = 2.0;

% Set the initial temperature to be the measured initial temperature
t0 = Exp_data.(exp_id).InsideTemp.values(1);

% 
power = [tin Heating_coeff .* (Exp_data.(exp_id).Power.values - 1.67 * 1000)];

% Turn down the air exchange rate when the HVAC is not running
night_air_exchange_rate = 0;
air_exchange_rate(abs(power(:, 2)) < 100, 2) = night_air_exchange_rate;


%% Run the simulation
% Note: The simlulink model loads the data separately, includes the
% calculated solar position and radiations from pvlib
load_system("polydome");
set_param('polydome', 'StopTime', int2str(tin(end)));
simout = sim("polydome");

%% Compare the simulation results with the measured values
SimulatedTemp = simout.SimulatedTemp.Data;

figure; hold on; grid minor;
plot(tin, InsideTemp);
plot(tin, OutsideTemp);
plot(simout.tout, SimulatedTemp, 'LineWidth', 2);


legend('InsideTemp', 'OutsideTemp', 'SimulatedTemp');

x0=500;
y0=300;
width=1500;
height=500;
set(gcf,'position',[x0,y0,width,height]);
title(exp_id);
%title(sprintf('Night Air exchange rate %f', night_air_exchange_rate));

hold off;

saveas(gcf, strcat(exp_id, '_simulation'), 'svg')
