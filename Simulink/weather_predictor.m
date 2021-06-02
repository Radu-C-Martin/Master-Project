classdef weather_predictor < matlab.System
    % untitled Add summary here
    %
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.

    % Public, tunable properties
    properties
        
    end
    
    % Public, tunable properties
    properties(Nontunable)
        TimeStep = 0;
        N = 0;
    end

    properties(DiscreteState)

    end

    % Pre-computed constants
    properties(Access = private)
        
    end

    methods(Access = protected)      
        function num = getNumInputsImpl(~)
            num = 2;
        end
        function num = getNumOutputsImpl(~)
            num = 2;
        end
        function [dt1, dt2] = getOutputDataTypeImpl(~)
        	dt1 = 'double';
            dt2 = 'double';

        end
        function [dt1, dt2] = getInputDataTypeImpl(~)
        	dt1 = 'double';
            dt2 = 'double';
        end
        function [sz1, sz2] = getOutputSizeImpl(obj)
            sz1 = [1 2];
        	sz2 = [obj.N 2];
        end
        function sz1 = getInputSizeImpl(~)
        	sz1 = 1;
        end
        function cp1 = isInputComplexImpl(~)
        	cp1 = false;
        end
        function [cp1, cp2] = isOutputComplexImpl(~)
        	cp1 = false;
            cp2 = false;
        end
        function fz1 = isInputFixedSizeImpl(~)
            fz1 = true;
        end
        function [fz1, fz2] = isOutputFixedSizeImpl(~)
        	fz1 = true;
            fz2 = true;
        end
        
        
        function setupImpl(~, ~, ~)
            disp('Hello World')
            % Perform one-time calculations, such as computing constants
        end

        function [w, w_hat] = stepImpl(obj,wdb_mat,timestamp)
            disp(timestamp)
            
            
            forecast_start = timestamp + obj.TimeStep;
            forecast_end = timestamp + obj.N * obj.TimeStep;
            
            xq = forecast_start:obj.TimeStep:forecast_end;
            
            weather_start_idx = find(wdb_mat(:, 1) <= timestamp, 1);
            weather_end_idx = find(wdb_mat(:, 1) >= forecast_end, 1);
            weather_idx = weather_start_idx:weather_end_idx;
            
            solar_direct = interp1(wdb_mat(weather_idx, 1), wdb_mat(weather_idx, 18), timestamp);
            solar_diffuse = interp1(wdb_mat(weather_idx, 1), wdb_mat(weather_idx, 19), timestamp);
            outside_temp = interp1(wdb_mat(weather_idx, 1), wdb_mat(weather_idx, 7), timestamp);
            w = [solar_direct + solar_diffuse, outside_temp];
            
            solar_direct = interp1(wdb_mat(weather_idx, 1), wdb_mat(weather_idx, 18), xq)';
            solar_diffuse = interp1(wdb_mat(weather_idx, 1), wdb_mat(weather_idx, 19), xq)';
            outside_temp = interp1(wdb_mat(weather_idx, 1), wdb_mat(weather_idx, 7), xq)';
            
            w_hat = [solar_direct + solar_diffuse, outside_temp];
        end

        function resetImpl(obj)
            % Initialize / reset discrete-state properties
        end
    end
end
