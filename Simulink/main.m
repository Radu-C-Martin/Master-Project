clear all
close all
clc
%%%%%%%%%%%%%%%%%%%%%%%


%% Set the model parameters

% Large side windows
window_size = [2.5 25];
window_roof_size = [5 5];
surface_part = 0.1;
U = 1.8; % heat transfer coefficient [W/m2K]
g = 0.7; % total solar energy transmittance
v_g = 0.65; % transmittance in visible range of the sunlight

% Roof
wall_size = [25 25];
roof_position = [0 0 0];
% The roof is supposed to be made of [5cm wood, 10cm insulation, 5cm wood]
node_thickness = [0.05 0.10 0.05]; % Data from 03.03 email with Manuel
% Data from https://simulationresearch.lbl.gov/modelica/releases/latest/help
% /Buildings_HeatTransfer_Data_Solids.html#Buildings.HeatTransfer.Data.Solids.Plywood
node_conductivity = [0.12 0.03 0.12]; 
node_capacity = [1210 1200 1210];
node_density = [540 40 540];

% Floor
ceiling_size = [25 25];
% The floor is supposed to be made of []
layer_thickness = [0.05 0.10 0.20];
layer_conductivity = [0.12 0.03 1.4];
layer_capacity = [1210 1200 840];
layer_density = [540 40 2240];



%% Set the run parameters

air_exchange_rate = 2.0;
power = 0;

Te = 60*60*24*365;
sim("polydome_model_1")
