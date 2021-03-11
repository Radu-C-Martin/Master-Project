clear all
close all
clc
%%%%%%%%%%%%%%%%%%%%%%%

%% Load the experimental data
Exp1 = load("../Data/Luca_experimental_data/Exp1.mat");
py_Exp1 = load("../Data/Exp1_WDB.mat");

tin = py_Exp1.Exp1_WDB(:,1);

% The power trick: when the setpoint is larger than the actual temperature
% the HVAC system is heating the room, otherwise it is cooling the room
Setpoint = Exp1.Exp1.Setpoint.values;
InsideTemp = mean([Exp1.Exp1.InsideTemp.values, Exp1.Exp1.LakeTemp.values], 2);
OutsideTemp = Exp1.Exp1.OutsideTemp.values;
HVAC_COP = 4.5;
Heating_coeff = sign(Setpoint - InsideTemp);
Heating_coeff(Heating_coeff == -1) = -1 * HVAC_COP;


%% Set the model parameters

% Large side windows
window_size = [2 25];
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

air_exchange_rate = tin;
air_exchange_rate(:,2) = 2.0;
t0 = 24;

power = [tin Heating_coeff .* (Exp1.Exp1.Power.values - 1.67 * 1000)];

Te = 60*60*24*365;

%% Run the simulation
% Note: The simlulink model loads the data separately, includes the
% calculated solar position and radiations from pvlib
simout = sim("polydome_model_1");

%% Compare the simulation results with the measured values
SimulatedTemp = simout.SimulatedTemp.Data;

figure; hold on; grid minor;
plot(tin, InsideTemp);
plot(tin, OutsideTemp);
plot(simout.tout, SimulatedTemp, 'LineWidth', 2);
plot(tin, Setpoint);
legend('InsideTemp', 'OutsideTemp', 'SimulatedTemp', 'Setpoint');
hold off;

% calculation notes for furniture wall parameters

% surface:
% 1/4 * 1.8 [m2/m2 of floor space] * 625 m2 surface = 140 m2
% 140 m2 = [7 20] m [height width]

% mass:
% 1/4 * 40 [kg/m2 of floor space] * 625 m2 surface = 6250 kg

% volume:
% 6250[kg]/600[kg/m3] = 10.41 [m3]

% thickness:
%10.41[m3]/140[m2] = 0.075m = 7.5cm

