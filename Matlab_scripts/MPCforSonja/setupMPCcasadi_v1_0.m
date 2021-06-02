

%% Settings
TimeStep = 900; % Step time
nHor     = 4*24; % Length of ontrol and planning horizon
%tSmp     = 0:TimeStep:nHor*TimeStep-1;

nStt     = 1; % Number of states
chY      = 1; % Number of observed variables
nDst     = 1; % Number of disturbance variables
nMV      = 1; % Number of controlled variables

%% System matrices
A  = 1;
B  = [-1, 1]/(3000*4182/TimeStep);
Bd = B(:, 1:nDst);
Bu = B(:, nDst+1:end);
C  = 1;
D  = 0;

%% Constraints and normalization
uMin = 0; 
uMax = 7500; 
yMin = 40;
yMax = 50;

%% Weights
R   = 1/uMax/0.1; 
T   = 1e5*eye(chY); 

