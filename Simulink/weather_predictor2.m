function w = weather_predictor2(wdb_mat,timestamp, N)
%WEATHER_PREDICTOR2 Summary of this function goes here
%   Detailed explanation goes here
curr_idx = find(wdb_mat(:, 1) == timestamp);
N_idx = (1:N) + curr_idx;
w = [wdb_mat(N_idx, 18) + wdb_mat(N_idx, 19), wdb_mat(N_idx, 7)];

end