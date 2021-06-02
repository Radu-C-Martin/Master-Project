classdef gp_mpc_system < matlab.System & matlab.system.mixin.Propagates
    % untitled Add summary here
    %
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.

    properties
        % Control horizon
        N = 0; 
        % Time Step
        TimeStep = 0;
        % Max Electrical Power Consumption
        Pel = 6300;

    end

    properties (DiscreteState)
    end

    properties (Access = private)
        % Pre-computed constants.
        casadi_solver
        u_lags
        y_lags
        lbx
        ubx
    end

    methods (Access = protected)
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
        function sz1 = getOutputSizeImpl(~)
        	sz1 = 1;
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
        function setupImpl(obj,~,~)
            % Implement tasks that need to be performed only once, 
            % such as pre-computed constants.
            addpath('/home/radu/Media/MATLAB/casadi-linux-matlabR2014b-v3.5.5')
            import casadi.*
            
            % Initialize CasADi callback
            cs_model = gpCallback('model');
            
            % Set up problem variables
            T_set = 20;
            n_states = 7;
            
            COP = 5; %cooling
            EER = 5; %heating
            
            
            obj.u_lags = [0];
            obj.y_lags = [23 23 23];
            
            % Formulate the optimization problem
            J = 0; % optimization objective
            g = []; % constraints vector

            % Set up the symbolic variables
            U = MX.sym('U', obj.N, 1);
            W = MX.sym('W', obj.N, 2);
            x0 = MX.sym('x0', 1, n_states - 3);

            % setup the first state
            wk = W(1, :);
            uk = U(1); % scaled input
            xk = [wk, obj.Pel*uk, x0];
            yk = cs_model(xk);
            J = J + (yk - T_set).^2;
            
            % Setup the rest of the states
            for idx = 2:obj.N
                wk = W(idx, :); 
                uk_1 = uk; uk = U(idx);
                xk = [wk, obj.Pel*uk, obj.Pel*uk_1, yk, xk(5:6)];
                yk = cs_model(xk);
                J = J + (yk - T_set).^2;
            end
            
            p = [vec(W); vec(x0)];
            nlp_prob = struct('f', J, 'x', vec(U), 'g', g, 'p', p);
            
            
            opts = struct;
            %opts.ipopt.max_iter = 5000;
            opts.ipopt.max_cpu_time = 15 * 60;
            opts.ipopt.hessian_approximation = 'limited-memory';
            %opts.ipopt.print_level =0;%0,3
            opts.print_time = 0;
            opts.ipopt.acceptable_tol =1e-8;
            opts.ipopt.acceptable_obj_change_tol = 1e-6;
            
            solver = nlpsol('solver', 'ipopt', nlp_prob,opts);
            
            obj.casadi_solver = solver;
            obj.lbx = -COP;
            obj.ubx = EER;
        end

        function u = stepImpl(obj,x,w)        
            import casadi.*
            
            %Update the y lags
            obj.y_lags = [x, obj.y_lags(1:end-1)];
            
            
            real_p = vertcat(vec(DM(w)), vec(DM([obj.u_lags obj.y_lags])));
            disp("Starting optimization")
            tic
            res = obj.casadi_solver('p', real_p, 'ubx', obj.ubx, 'lbx', obj.lbx);
            t = toc;
            disp(t)
            u = obj.Pel * full(res.x(1));
            u = 15000 * (20 - x);
            
            % Update the u lags
            obj.u_lags = [u, obj.u_lags(2:end-1)];
            
           
            
        end

        function resetImpl(obj)
            % Initialize discrete-state properties.
        end
    end
end
