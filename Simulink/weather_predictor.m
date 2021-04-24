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
            num = 1;
        end
        function dt1 = getOutputDataTypeImpl(~)
        	dt1 = 'double';
        end
        function [dt1, dt2] = getInputDataTypeImpl(~)
        	dt1 = 'double';
            dt2 = 'double';
        end
        function sz1 = getOutputSizeImpl(obj)
        	sz1 = [obj.N 2];
        end
        function sz1 = getInputSizeImpl(~)
        	sz1 = 1;
        end
        function cp1 = isInputComplexImpl(~)
        	cp1 = false;
        end
        function cp1 = isOutputComplexImpl(~)
        	cp1 = false;
        end
        function fz1 = isInputFixedSizeImpl(~)
        	fz1 = true;
        end
        function fz1 = isOutputFixedSizeImpl(~)
        	fz1 = true;
        end
        
        
        function setupImpl(~, ~, ~)
            disp('Hello World')
            % Perform one-time calculations, such as computing constants
        end

        function w = stepImpl(obj,wdb_mat,timestamp)
            disp(timestamp)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            curr_idx = find(wdb_mat(:, 1) == timestamp);
            N_idx = (1:obj.N) + curr_idx;
            w = [wdb_mat(N_idx, 18) + wdb_mat(N_idx, 19), wdb_mat(N_idx, 7)];
        end

        function resetImpl(obj)
            % Initialize / reset discrete-state properties
        end
    end
end
